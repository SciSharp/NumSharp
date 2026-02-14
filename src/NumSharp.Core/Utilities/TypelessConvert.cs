using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public delegate object TypelessConvertDelegate(object input);

    /// <summary>
    ///     Provides a way to convert boxed object from known input type to known output type.
    ///     By making it receive and return <see cref="object"/> - It is suitable for a common delegate: see <see cref="TypelessConvertDelegate"/>
    /// </summary>
    public static class TypelessConvert
    {
        private static readonly FrozenDictionary<(Type input, Type output), TypelessConvertDelegate> _delegates;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        static TypelessConvert()
        {
            // ReSharper disable once UseObjectOrCollectionInitializer
            var delegates = new Dictionary<(Type input, Type output), TypelessConvertDelegate>();
#if _REGEN
            %foreach forevery(supported_primitives, supported_primitives, true)%
            delegates.Add((typeof(#1), typeof(#2)), From#1To#2);
            %
#else
            delegates.Add((typeof(Boolean), typeof(Byte)), FromBooleanToByte);
            delegates.Add((typeof(Boolean), typeof(Int16)), FromBooleanToInt16);
            delegates.Add((typeof(Boolean), typeof(UInt16)), FromBooleanToUInt16);
            delegates.Add((typeof(Boolean), typeof(Int32)), FromBooleanToInt32);
            delegates.Add((typeof(Boolean), typeof(UInt32)), FromBooleanToUInt32);
            delegates.Add((typeof(Boolean), typeof(Int64)), FromBooleanToInt64);
            delegates.Add((typeof(Boolean), typeof(UInt64)), FromBooleanToUInt64);
            delegates.Add((typeof(Boolean), typeof(Char)), FromBooleanToChar);
            delegates.Add((typeof(Boolean), typeof(Double)), FromBooleanToDouble);
            delegates.Add((typeof(Boolean), typeof(Single)), FromBooleanToSingle);
            delegates.Add((typeof(Boolean), typeof(Decimal)), FromBooleanToDecimal);
            delegates.Add((typeof(Boolean), typeof(String)), FromBooleanToString);
            delegates.Add((typeof(Byte), typeof(Boolean)), FromByteToBoolean);
            delegates.Add((typeof(Byte), typeof(Int16)), FromByteToInt16);
            delegates.Add((typeof(Byte), typeof(UInt16)), FromByteToUInt16);
            delegates.Add((typeof(Byte), typeof(Int32)), FromByteToInt32);
            delegates.Add((typeof(Byte), typeof(UInt32)), FromByteToUInt32);
            delegates.Add((typeof(Byte), typeof(Int64)), FromByteToInt64);
            delegates.Add((typeof(Byte), typeof(UInt64)), FromByteToUInt64);
            delegates.Add((typeof(Byte), typeof(Char)), FromByteToChar);
            delegates.Add((typeof(Byte), typeof(Double)), FromByteToDouble);
            delegates.Add((typeof(Byte), typeof(Single)), FromByteToSingle);
            delegates.Add((typeof(Byte), typeof(Decimal)), FromByteToDecimal);
            delegates.Add((typeof(Byte), typeof(String)), FromByteToString);
            delegates.Add((typeof(Int16), typeof(Boolean)), FromInt16ToBoolean);
            delegates.Add((typeof(Int16), typeof(Byte)), FromInt16ToByte);
            delegates.Add((typeof(Int16), typeof(UInt16)), FromInt16ToUInt16);
            delegates.Add((typeof(Int16), typeof(Int32)), FromInt16ToInt32);
            delegates.Add((typeof(Int16), typeof(UInt32)), FromInt16ToUInt32);
            delegates.Add((typeof(Int16), typeof(Int64)), FromInt16ToInt64);
            delegates.Add((typeof(Int16), typeof(UInt64)), FromInt16ToUInt64);
            delegates.Add((typeof(Int16), typeof(Char)), FromInt16ToChar);
            delegates.Add((typeof(Int16), typeof(Double)), FromInt16ToDouble);
            delegates.Add((typeof(Int16), typeof(Single)), FromInt16ToSingle);
            delegates.Add((typeof(Int16), typeof(Decimal)), FromInt16ToDecimal);
            delegates.Add((typeof(Int16), typeof(String)), FromInt16ToString);
            delegates.Add((typeof(UInt16), typeof(Boolean)), FromUInt16ToBoolean);
            delegates.Add((typeof(UInt16), typeof(Byte)), FromUInt16ToByte);
            delegates.Add((typeof(UInt16), typeof(Int16)), FromUInt16ToInt16);
            delegates.Add((typeof(UInt16), typeof(Int32)), FromUInt16ToInt32);
            delegates.Add((typeof(UInt16), typeof(UInt32)), FromUInt16ToUInt32);
            delegates.Add((typeof(UInt16), typeof(Int64)), FromUInt16ToInt64);
            delegates.Add((typeof(UInt16), typeof(UInt64)), FromUInt16ToUInt64);
            delegates.Add((typeof(UInt16), typeof(Char)), FromUInt16ToChar);
            delegates.Add((typeof(UInt16), typeof(Double)), FromUInt16ToDouble);
            delegates.Add((typeof(UInt16), typeof(Single)), FromUInt16ToSingle);
            delegates.Add((typeof(UInt16), typeof(Decimal)), FromUInt16ToDecimal);
            delegates.Add((typeof(UInt16), typeof(String)), FromUInt16ToString);
            delegates.Add((typeof(Int32), typeof(Boolean)), FromInt32ToBoolean);
            delegates.Add((typeof(Int32), typeof(Byte)), FromInt32ToByte);
            delegates.Add((typeof(Int32), typeof(Int16)), FromInt32ToInt16);
            delegates.Add((typeof(Int32), typeof(UInt16)), FromInt32ToUInt16);
            delegates.Add((typeof(Int32), typeof(UInt32)), FromInt32ToUInt32);
            delegates.Add((typeof(Int32), typeof(Int64)), FromInt32ToInt64);
            delegates.Add((typeof(Int32), typeof(UInt64)), FromInt32ToUInt64);
            delegates.Add((typeof(Int32), typeof(Char)), FromInt32ToChar);
            delegates.Add((typeof(Int32), typeof(Double)), FromInt32ToDouble);
            delegates.Add((typeof(Int32), typeof(Single)), FromInt32ToSingle);
            delegates.Add((typeof(Int32), typeof(Decimal)), FromInt32ToDecimal);
            delegates.Add((typeof(Int32), typeof(String)), FromInt32ToString);
            delegates.Add((typeof(UInt32), typeof(Boolean)), FromUInt32ToBoolean);
            delegates.Add((typeof(UInt32), typeof(Byte)), FromUInt32ToByte);
            delegates.Add((typeof(UInt32), typeof(Int16)), FromUInt32ToInt16);
            delegates.Add((typeof(UInt32), typeof(UInt16)), FromUInt32ToUInt16);
            delegates.Add((typeof(UInt32), typeof(Int32)), FromUInt32ToInt32);
            delegates.Add((typeof(UInt32), typeof(Int64)), FromUInt32ToInt64);
            delegates.Add((typeof(UInt32), typeof(UInt64)), FromUInt32ToUInt64);
            delegates.Add((typeof(UInt32), typeof(Char)), FromUInt32ToChar);
            delegates.Add((typeof(UInt32), typeof(Double)), FromUInt32ToDouble);
            delegates.Add((typeof(UInt32), typeof(Single)), FromUInt32ToSingle);
            delegates.Add((typeof(UInt32), typeof(Decimal)), FromUInt32ToDecimal);
            delegates.Add((typeof(UInt32), typeof(String)), FromUInt32ToString);
            delegates.Add((typeof(Int64), typeof(Boolean)), FromInt64ToBoolean);
            delegates.Add((typeof(Int64), typeof(Byte)), FromInt64ToByte);
            delegates.Add((typeof(Int64), typeof(Int16)), FromInt64ToInt16);
            delegates.Add((typeof(Int64), typeof(UInt16)), FromInt64ToUInt16);
            delegates.Add((typeof(Int64), typeof(Int32)), FromInt64ToInt32);
            delegates.Add((typeof(Int64), typeof(UInt32)), FromInt64ToUInt32);
            delegates.Add((typeof(Int64), typeof(UInt64)), FromInt64ToUInt64);
            delegates.Add((typeof(Int64), typeof(Char)), FromInt64ToChar);
            delegates.Add((typeof(Int64), typeof(Double)), FromInt64ToDouble);
            delegates.Add((typeof(Int64), typeof(Single)), FromInt64ToSingle);
            delegates.Add((typeof(Int64), typeof(Decimal)), FromInt64ToDecimal);
            delegates.Add((typeof(Int64), typeof(String)), FromInt64ToString);
            delegates.Add((typeof(UInt64), typeof(Boolean)), FromUInt64ToBoolean);
            delegates.Add((typeof(UInt64), typeof(Byte)), FromUInt64ToByte);
            delegates.Add((typeof(UInt64), typeof(Int16)), FromUInt64ToInt16);
            delegates.Add((typeof(UInt64), typeof(UInt16)), FromUInt64ToUInt16);
            delegates.Add((typeof(UInt64), typeof(Int32)), FromUInt64ToInt32);
            delegates.Add((typeof(UInt64), typeof(UInt32)), FromUInt64ToUInt32);
            delegates.Add((typeof(UInt64), typeof(Int64)), FromUInt64ToInt64);
            delegates.Add((typeof(UInt64), typeof(Char)), FromUInt64ToChar);
            delegates.Add((typeof(UInt64), typeof(Double)), FromUInt64ToDouble);
            delegates.Add((typeof(UInt64), typeof(Single)), FromUInt64ToSingle);
            delegates.Add((typeof(UInt64), typeof(Decimal)), FromUInt64ToDecimal);
            delegates.Add((typeof(UInt64), typeof(String)), FromUInt64ToString);
            delegates.Add((typeof(Char), typeof(Boolean)), FromCharToBoolean);
            delegates.Add((typeof(Char), typeof(Byte)), FromCharToByte);
            delegates.Add((typeof(Char), typeof(Int16)), FromCharToInt16);
            delegates.Add((typeof(Char), typeof(UInt16)), FromCharToUInt16);
            delegates.Add((typeof(Char), typeof(Int32)), FromCharToInt32);
            delegates.Add((typeof(Char), typeof(UInt32)), FromCharToUInt32);
            delegates.Add((typeof(Char), typeof(Int64)), FromCharToInt64);
            delegates.Add((typeof(Char), typeof(UInt64)), FromCharToUInt64);
            delegates.Add((typeof(Char), typeof(Double)), FromCharToDouble);
            delegates.Add((typeof(Char), typeof(Single)), FromCharToSingle);
            delegates.Add((typeof(Char), typeof(Decimal)), FromCharToDecimal);
            delegates.Add((typeof(Char), typeof(String)), FromCharToString);
            delegates.Add((typeof(Double), typeof(Boolean)), FromDoubleToBoolean);
            delegates.Add((typeof(Double), typeof(Byte)), FromDoubleToByte);
            delegates.Add((typeof(Double), typeof(Int16)), FromDoubleToInt16);
            delegates.Add((typeof(Double), typeof(UInt16)), FromDoubleToUInt16);
            delegates.Add((typeof(Double), typeof(Int32)), FromDoubleToInt32);
            delegates.Add((typeof(Double), typeof(UInt32)), FromDoubleToUInt32);
            delegates.Add((typeof(Double), typeof(Int64)), FromDoubleToInt64);
            delegates.Add((typeof(Double), typeof(UInt64)), FromDoubleToUInt64);
            delegates.Add((typeof(Double), typeof(Char)), FromDoubleToChar);
            delegates.Add((typeof(Double), typeof(Single)), FromDoubleToSingle);
            delegates.Add((typeof(Double), typeof(Decimal)), FromDoubleToDecimal);
            delegates.Add((typeof(Double), typeof(String)), FromDoubleToString);
            delegates.Add((typeof(Single), typeof(Boolean)), FromSingleToBoolean);
            delegates.Add((typeof(Single), typeof(Byte)), FromSingleToByte);
            delegates.Add((typeof(Single), typeof(Int16)), FromSingleToInt16);
            delegates.Add((typeof(Single), typeof(UInt16)), FromSingleToUInt16);
            delegates.Add((typeof(Single), typeof(Int32)), FromSingleToInt32);
            delegates.Add((typeof(Single), typeof(UInt32)), FromSingleToUInt32);
            delegates.Add((typeof(Single), typeof(Int64)), FromSingleToInt64);
            delegates.Add((typeof(Single), typeof(UInt64)), FromSingleToUInt64);
            delegates.Add((typeof(Single), typeof(Char)), FromSingleToChar);
            delegates.Add((typeof(Single), typeof(Double)), FromSingleToDouble);
            delegates.Add((typeof(Single), typeof(Decimal)), FromSingleToDecimal);
            delegates.Add((typeof(Single), typeof(String)), FromSingleToString);
            delegates.Add((typeof(Decimal), typeof(Boolean)), FromDecimalToBoolean);
            delegates.Add((typeof(Decimal), typeof(Byte)), FromDecimalToByte);
            delegates.Add((typeof(Decimal), typeof(Int16)), FromDecimalToInt16);
            delegates.Add((typeof(Decimal), typeof(UInt16)), FromDecimalToUInt16);
            delegates.Add((typeof(Decimal), typeof(Int32)), FromDecimalToInt32);
            delegates.Add((typeof(Decimal), typeof(UInt32)), FromDecimalToUInt32);
            delegates.Add((typeof(Decimal), typeof(Int64)), FromDecimalToInt64);
            delegates.Add((typeof(Decimal), typeof(UInt64)), FromDecimalToUInt64);
            delegates.Add((typeof(Decimal), typeof(Char)), FromDecimalToChar);
            delegates.Add((typeof(Decimal), typeof(Double)), FromDecimalToDouble);
            delegates.Add((typeof(Decimal), typeof(Single)), FromDecimalToSingle);
            delegates.Add((typeof(Decimal), typeof(String)), FromDecimalToString);
            delegates.Add((typeof(String), typeof(Boolean)), FromStringToBoolean);
            delegates.Add((typeof(String), typeof(Byte)), FromStringToByte);
            delegates.Add((typeof(String), typeof(Int16)), FromStringToInt16);
            delegates.Add((typeof(String), typeof(UInt16)), FromStringToUInt16);
            delegates.Add((typeof(String), typeof(Int32)), FromStringToInt32);
            delegates.Add((typeof(String), typeof(UInt32)), FromStringToUInt32);
            delegates.Add((typeof(String), typeof(Int64)), FromStringToInt64);
            delegates.Add((typeof(String), typeof(UInt64)), FromStringToUInt64);
            delegates.Add((typeof(String), typeof(Char)), FromStringToChar);
            delegates.Add((typeof(String), typeof(Double)), FromStringToDouble);
            delegates.Add((typeof(String), typeof(Single)), FromStringToSingle);
            delegates.Add((typeof(String), typeof(Decimal)), FromStringToDecimal);
#endif
            _delegates = delegates.ToFrozenDictionary();
        }

        public static TypelessConvertDelegate GetConverter(Type input, Type output)
        {
            return _delegates[(input, output)];
        }

#if _REGEN
        %foreach forevery(supported_primitives, supported_primitives, true)%
        /// <summary>
        ///     Convert from #1 to #2 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="#1"/> and then converted to <see cref="#2"/></param>
        /// <returns>#2</returns>
        [MethodImpl(Inline)]
        public static object From#1To#2(object input)
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
        [MethodImpl(Inline)]
        public static object FromBooleanToByte(object input)
        {
            return Converts.ToByte((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToInt16(object input)
        {
            return Converts.ToInt16((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToUInt16(object input)
        {
            return Converts.ToUInt16((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToInt32(object input)
        {
            return Converts.ToInt32((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToUInt32(object input)
        {
            return Converts.ToUInt32((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToInt64(object input)
        {
            return Converts.ToInt64((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToUInt64(object input)
        {
            return Converts.ToUInt64((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToChar(object input)
        {
            return Converts.ToChar((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToDouble(object input)
        {
            return Converts.ToDouble((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToSingle(object input)
        {
            return Converts.ToSingle((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToDecimal(object input)
        {
            return Converts.ToDecimal((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromBooleanToString(object input)
        {
            return Converts.ToString((Boolean)input);
        }

        /// <summary>
        ///     Convert from Byte to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromByteToBoolean(object input)
        {
            return Converts.ToBoolean((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromByteToInt16(object input)
        {
            return Converts.ToInt16((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromByteToUInt16(object input)
        {
            return Converts.ToUInt16((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromByteToInt32(object input)
        {
            return Converts.ToInt32((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromByteToUInt32(object input)
        {
            return Converts.ToUInt32((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromByteToInt64(object input)
        {
            return Converts.ToInt64((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromByteToUInt64(object input)
        {
            return Converts.ToUInt64((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromByteToChar(object input)
        {
            return Converts.ToChar((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromByteToDouble(object input)
        {
            return Converts.ToDouble((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromByteToSingle(object input)
        {
            return Converts.ToSingle((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromByteToDecimal(object input)
        {
            return Converts.ToDecimal((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromByteToString(object input)
        {
            return Converts.ToString((Byte)input);
        }

        /// <summary>
        ///     Convert from Int16 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToBoolean(object input)
        {
            return Converts.ToBoolean((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToByte(object input)
        {
            return Converts.ToByte((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToUInt16(object input)
        {
            return Converts.ToUInt16((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToInt32(object input)
        {
            return Converts.ToInt32((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToUInt32(object input)
        {
            return Converts.ToUInt32((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToInt64(object input)
        {
            return Converts.ToInt64((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToUInt64(object input)
        {
            return Converts.ToUInt64((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToChar(object input)
        {
            return Converts.ToChar((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToDouble(object input)
        {
            return Converts.ToDouble((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToSingle(object input)
        {
            return Converts.ToSingle((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToDecimal(object input)
        {
            return Converts.ToDecimal((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromInt16ToString(object input)
        {
            return Converts.ToString((Int16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToBoolean(object input)
        {
            return Converts.ToBoolean((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToByte(object input)
        {
            return Converts.ToByte((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToInt16(object input)
        {
            return Converts.ToInt16((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToInt32(object input)
        {
            return Converts.ToInt32((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToUInt32(object input)
        {
            return Converts.ToUInt32((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToInt64(object input)
        {
            return Converts.ToInt64((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToUInt64(object input)
        {
            return Converts.ToUInt64((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToChar(object input)
        {
            return Converts.ToChar((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToDouble(object input)
        {
            return Converts.ToDouble((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToSingle(object input)
        {
            return Converts.ToSingle((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToDecimal(object input)
        {
            return Converts.ToDecimal((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromUInt16ToString(object input)
        {
            return Converts.ToString((UInt16)input);
        }

        /// <summary>
        ///     Convert from Int32 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToBoolean(object input)
        {
            return Converts.ToBoolean((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToByte(object input)
        {
            return Converts.ToByte((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToInt16(object input)
        {
            return Converts.ToInt16((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToUInt16(object input)
        {
            return Converts.ToUInt16((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToUInt32(object input)
        {
            return Converts.ToUInt32((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToInt64(object input)
        {
            return Converts.ToInt64((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToUInt64(object input)
        {
            return Converts.ToUInt64((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToChar(object input)
        {
            return Converts.ToChar((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToDouble(object input)
        {
            return Converts.ToDouble((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToSingle(object input)
        {
            return Converts.ToSingle((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToDecimal(object input)
        {
            return Converts.ToDecimal((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromInt32ToString(object input)
        {
            return Converts.ToString((Int32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToBoolean(object input)
        {
            return Converts.ToBoolean((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToByte(object input)
        {
            return Converts.ToByte((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToInt16(object input)
        {
            return Converts.ToInt16((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToUInt16(object input)
        {
            return Converts.ToUInt16((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToInt32(object input)
        {
            return Converts.ToInt32((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToInt64(object input)
        {
            return Converts.ToInt64((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToUInt64(object input)
        {
            return Converts.ToUInt64((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToChar(object input)
        {
            return Converts.ToChar((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToDouble(object input)
        {
            return Converts.ToDouble((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToSingle(object input)
        {
            return Converts.ToSingle((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToDecimal(object input)
        {
            return Converts.ToDecimal((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromUInt32ToString(object input)
        {
            return Converts.ToString((UInt32)input);
        }

        /// <summary>
        ///     Convert from Int64 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToBoolean(object input)
        {
            return Converts.ToBoolean((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToByte(object input)
        {
            return Converts.ToByte((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToInt16(object input)
        {
            return Converts.ToInt16((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToUInt16(object input)
        {
            return Converts.ToUInt16((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToInt32(object input)
        {
            return Converts.ToInt32((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToUInt32(object input)
        {
            return Converts.ToUInt32((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToUInt64(object input)
        {
            return Converts.ToUInt64((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToChar(object input)
        {
            return Converts.ToChar((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToDouble(object input)
        {
            return Converts.ToDouble((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToSingle(object input)
        {
            return Converts.ToSingle((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToDecimal(object input)
        {
            return Converts.ToDecimal((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromInt64ToString(object input)
        {
            return Converts.ToString((Int64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToBoolean(object input)
        {
            return Converts.ToBoolean((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToByte(object input)
        {
            return Converts.ToByte((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToInt16(object input)
        {
            return Converts.ToInt16((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToUInt16(object input)
        {
            return Converts.ToUInt16((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToInt32(object input)
        {
            return Converts.ToInt32((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToUInt32(object input)
        {
            return Converts.ToUInt32((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToInt64(object input)
        {
            return Converts.ToInt64((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToChar(object input)
        {
            return Converts.ToChar((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToDouble(object input)
        {
            return Converts.ToDouble((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToSingle(object input)
        {
            return Converts.ToSingle((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToDecimal(object input)
        {
            return Converts.ToDecimal((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromUInt64ToString(object input)
        {
            return Converts.ToString((UInt64)input);
        }

        /// <summary>
        ///     Convert from Char to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromCharToBoolean(object input)
        {
            return Converts.ToBoolean((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromCharToByte(object input)
        {
            return Converts.ToByte((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromCharToInt16(object input)
        {
            return Converts.ToInt16((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromCharToUInt16(object input)
        {
            return Converts.ToUInt16((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromCharToInt32(object input)
        {
            return Converts.ToInt32((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromCharToUInt32(object input)
        {
            return Converts.ToUInt32((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromCharToInt64(object input)
        {
            return Converts.ToInt64((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromCharToUInt64(object input)
        {
            return Converts.ToUInt64((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromCharToDouble(object input)
        {
            return Converts.ToDouble((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromCharToSingle(object input)
        {
            return Converts.ToSingle((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromCharToDecimal(object input)
        {
            return Converts.ToDecimal((Char)input);
        }

        /// <summary>
        ///     Convert from Char to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromCharToString(object input)
        {
            return Converts.ToString((Char)input);
        }

        /// <summary>
        ///     Convert from Double to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToBoolean(object input)
        {
            return Converts.ToBoolean((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToByte(object input)
        {
            return Converts.ToByte((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToInt16(object input)
        {
            return Converts.ToInt16((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToUInt16(object input)
        {
            return Converts.ToUInt16((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToInt32(object input)
        {
            return Converts.ToInt32((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToUInt32(object input)
        {
            return Converts.ToUInt32((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToInt64(object input)
        {
            return Converts.ToInt64((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToUInt64(object input)
        {
            return Converts.ToUInt64((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToChar(object input)
        {
            return Converts.ToChar((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToSingle(object input)
        {
            return Converts.ToSingle((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToDecimal(object input)
        {
            return Converts.ToDecimal((Double)input);
        }

        /// <summary>
        ///     Convert from Double to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromDoubleToString(object input)
        {
            return Converts.ToString((Double)input);
        }

        /// <summary>
        ///     Convert from Single to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToBoolean(object input)
        {
            return Converts.ToBoolean((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToByte(object input)
        {
            return Converts.ToByte((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToInt16(object input)
        {
            return Converts.ToInt16((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToUInt16(object input)
        {
            return Converts.ToUInt16((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToInt32(object input)
        {
            return Converts.ToInt32((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToUInt32(object input)
        {
            return Converts.ToUInt32((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToInt64(object input)
        {
            return Converts.ToInt64((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToUInt64(object input)
        {
            return Converts.ToUInt64((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToChar(object input)
        {
            return Converts.ToChar((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToDouble(object input)
        {
            return Converts.ToDouble((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToDecimal(object input)
        {
            return Converts.ToDecimal((Single)input);
        }

        /// <summary>
        ///     Convert from Single to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromSingleToString(object input)
        {
            return Converts.ToString((Single)input);
        }

        /// <summary>
        ///     Convert from Decimal to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToBoolean(object input)
        {
            return Converts.ToBoolean((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToByte(object input)
        {
            return Converts.ToByte((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToInt16(object input)
        {
            return Converts.ToInt16((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToUInt16(object input)
        {
            return Converts.ToUInt16((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToInt32(object input)
        {
            return Converts.ToInt32((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToUInt32(object input)
        {
            return Converts.ToUInt32((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToInt64(object input)
        {
            return Converts.ToInt64((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToUInt64(object input)
        {
            return Converts.ToUInt64((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToChar(object input)
        {
            return Converts.ToChar((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToDouble(object input)
        {
            return Converts.ToDouble((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToSingle(object input)
        {
            return Converts.ToSingle((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(Inline)]
        public static object FromDecimalToString(object input)
        {
            return Converts.ToString((Decimal)input);
        }

        /// <summary>
        ///     Convert from String to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(Inline)]
        public static object FromStringToBoolean(object input)
        {
            return Converts.ToBoolean((String)input);
        }

        /// <summary>
        ///     Convert from String to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(Inline)]
        public static object FromStringToByte(object input)
        {
            return Converts.ToByte((String)input);
        }

        /// <summary>
        ///     Convert from String to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(Inline)]
        public static object FromStringToInt16(object input)
        {
            return Converts.ToInt16((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(Inline)]
        public static object FromStringToUInt16(object input)
        {
            return Converts.ToUInt16((String)input);
        }

        /// <summary>
        ///     Convert from String to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(Inline)]
        public static object FromStringToInt32(object input)
        {
            return Converts.ToInt32((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(Inline)]
        public static object FromStringToUInt32(object input)
        {
            return Converts.ToUInt32((String)input);
        }

        /// <summary>
        ///     Convert from String to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(Inline)]
        public static object FromStringToInt64(object input)
        {
            return Converts.ToInt64((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(Inline)]
        public static object FromStringToUInt64(object input)
        {
            return Converts.ToUInt64((String)input);
        }

        /// <summary>
        ///     Convert from String to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(Inline)]
        public static object FromStringToChar(object input)
        {
            return Converts.ToChar((String)input);
        }

        /// <summary>
        ///     Convert from String to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(Inline)]
        public static object FromStringToDouble(object input)
        {
            return Converts.ToDouble((String)input);
        }

        /// <summary>
        ///     Convert from String to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(Inline)]
        public static object FromStringToSingle(object input)
        {
            return Converts.ToSingle((String)input);
        }

        /// <summary>
        ///     Convert from String to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(Inline)]
        public static object FromStringToDecimal(object input)
        {
            return Converts.ToDecimal((String)input);
        }
#endif
    }
}
