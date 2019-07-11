using System;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    ///     Represents all available types in numpy.
    /// </summary>
    /// <remarks>The int values of the enum are a copy of <see cref="TypeCode"/> excluding types not available in numpy.</remarks>
    public enum NPTypeCode
    {
        /// <summary>A null reference.</summary>
        /// <returns></returns>
        Empty = 0,

        /// <summary>A simple type representing Boolean values of true or false.</summary>
        /// <returns></returns>
        Boolean = 3,

        /// <summary>An integral type representing unsigned 16-bit integers with values between 0 and 65535. The set of possible values for the <see cref="F:System.TypeCode.Char"></see> type corresponds to the Unicode character set.</summary>
        /// <returns></returns>
        Char = 4,

        /// <summary>An integral type representing unsigned 8-bit integers with values between 0 and 255.</summary>
        /// <returns></returns>
        Byte = 6,

        /// <summary>An integral type representing signed 16-bit integers with values between -32768 and 32767.</summary>
        /// <returns></returns>
        Int16 = 7,

        /// <summary>An integral type representing unsigned 16-bit integers with values between 0 and 65535.</summary>
        /// <returns></returns>
        UInt16 = 8,

        /// <summary>An integral type representing signed 32-bit integers with values between -2147483648 and 2147483647.</summary>
        /// <returns></returns>
        Int32 = 9,

        /// <summary>An integral type representing unsigned 32-bit integers with values between 0 and 4294967295.</summary>
        /// <returns></returns>
        UInt32 = 10, // 0x0000000A

        /// <summary>An integral type representing signed 64-bit integers with values between -9223372036854775808 and 9223372036854775807.</summary>
        /// <returns></returns>
        Int64 = 11, // 0x0000000B

        /// <summary>An integral type representing unsigned 64-bit integers with values between 0 and 18446744073709551615.</summary>
        /// <returns></returns>
        UInt64 = 12, // 0x0000000C

        /// <summary>A floating point type representing values ranging from approximately 1.5 x 10 -45 to 3.4 x 10 38 with a precision of 7 digits.</summary>
        /// <returns></returns>
        Single = 13, // 0x0000000D

        /// <summary>A floating point type representing values ranging from approximately 5.0 x 10 -324 to 1.7 x 10 308 with a precision of 15-16 digits.</summary>
        /// <returns></returns>
        Double = 14, // 0x0000000E

        /// <summary>A simple type representing values ranging from 1.0 x 10 -28 to approximately 7.9 x 10 28 with 28-29 significant digits.</summary>
        /// <returns></returns>
        Decimal = 15, // 0x0000000F

        /// <summary>A sealed class type representing Unicode character strings.</summary>
        /// <returns></returns>
        String = 18, // 0x00000012

        NDArray = 128, //0x00000080

        Complex = 129, //0x00000081
    }

    public static class NPTypeCodeExtensions
    {
        /// <summary>
        ///     Returns true if typecode is a number (incl. <see cref="bool"/>, <see cref="char"/> and <see cref="Complex"/>).
        /// </summary>
        public static bool IsNumerical(this NPTypeCode typeCode)
        {
            var val = (int)typeCode;
            return val >= 3 && val <= 15 || val == 129;
        }

        /// <summary>
        ///     Extracts <see cref="NPTypeCode"/> from given <see cref="Type"/>.
        /// </summary>
        /// <remarks>In case there was no successful cast to <see cref="NPTypeCode"/>, return will be <see cref="NPTypeCode.Empty"/></remarks>
        public static NPTypeCode GetTypeCode(this Type type)
        {
            // ReSharper disable once PossibleNullReferenceException
            while (type.IsArray)
                type = type.GetElementType();

            var tc = Type.GetTypeCode(type);
            if (tc == TypeCode.Object)
            {
                if (type == typeof(NDArray))
                {
                    return NPTypeCode.NDArray;
                }

                if (type == typeof(Complex))
                {
                    return NPTypeCode.Complex;
                }

                return NPTypeCode.Empty;
            }

            try
            {
                return (NPTypeCode)(int)tc;
            }
            catch (InvalidCastException)
            {
                return NPTypeCode.Empty;
            }
        }

        /// <summary>
        ///     Extracts <see cref="NPTypeCode"/> from given <typeparamref name="T"/>.
        /// </summary>
        /// <remarks>In case there was no successful cast to <see cref="NPTypeCode"/>, return will be <see cref="NPTypeCode.Empty"/></remarks>
        public static NPTypeCode GetTypeCode<T>()
        {
            return InfoOf<T>.NPTypeCode;
        }

        /// <summary>
        ///     Convert <see cref="NPTypeCode"/> into its <see cref="Type"/>
        /// </summary>
        /// <param name="typeCode"></param>
        /// <returns></returns>
        public static Type AsType(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Empty:
                    return null;
                case NPTypeCode.Boolean:
                    return typeof(Boolean);
                case NPTypeCode.Char:
                    return typeof(Char);
                case NPTypeCode.Byte:
                    return typeof(Byte);
                case NPTypeCode.Int16:
                    return typeof(Int16);
                case NPTypeCode.UInt16:
                    return typeof(UInt16);
                case NPTypeCode.Int32:
                    return typeof(Int32);
                case NPTypeCode.UInt32:
                    return typeof(UInt32);
                case NPTypeCode.Int64:
                    return typeof(Int64);
                case NPTypeCode.UInt64:
                    return typeof(UInt64);
                case NPTypeCode.Single:
                    return typeof(Single);
                case NPTypeCode.Double:
                    return typeof(Double);
                case NPTypeCode.Decimal:
                    return typeof(Decimal);
                case NPTypeCode.String:
                    return typeof(String);
                case NPTypeCode.NDArray:
                    return typeof(NDArray);
                case NPTypeCode.Complex:
                    return typeof(Complex);
                default:
                    throw new ArgumentOutOfRangeException(nameof(typeCode), typeCode, null);
            }
        }

        /// <summary>
        ///     Checks if given <see cref="Type"/> has a match in <see cref="NPTypeCode"/>.
        /// </summary>
        public static bool IsValidNPType(this Type type)
        {
            return type.GetTypeCode() != NPTypeCode.Empty;
        }
    }
}
