using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    public static class NumberInfo
    {
        /// <summary>
        ///     Get the min value of given <see cref="NPTypeCode"/>.
        /// </summary>
        public static object MaxValue(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Complex:
                    return new Complex(double.MaxValue, double.MaxValue);
                case NPTypeCode.Boolean:
                    return true;
#if _REGEN
	            %foreach except(supported_primitives, "Boolean", "String")%
                case NPTypeCode.#1:
                    return #1.MaxValue;
                %
#else
                case NPTypeCode.SByte:
                    return SByte.MaxValue;
                case NPTypeCode.Byte:
                    return Byte.MaxValue;
                case NPTypeCode.Int16:
                    return Int16.MaxValue;
                case NPTypeCode.UInt16:
                    return UInt16.MaxValue;
                case NPTypeCode.Int32:
                    return Int32.MaxValue;
                case NPTypeCode.UInt32:
                    return UInt32.MaxValue;
                case NPTypeCode.Int64:
                    return Int64.MaxValue;
                case NPTypeCode.UInt64:
                    return UInt64.MaxValue;
                case NPTypeCode.Char:
                    return Char.MaxValue;
                case NPTypeCode.Double:
                    return Double.MaxValue;
                case NPTypeCode.Single:
                    return Single.MaxValue;
                case NPTypeCode.Decimal:
                    return Decimal.MaxValue;
#endif
                default:
                    throw new ArgumentOutOfRangeException(nameof(typeCode), typeCode, null);
            }
        }

        /// <summary>
        ///     Get the min value of given <see cref="NPTypeCode"/>.
        /// </summary>
        public static object MinValue(this NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Complex:
                    return new Complex(double.MinValue, double.MinValue);
                case NPTypeCode.Boolean:
                    return false;
#if _REGEN
	            %foreach except(supported_primitives, "Boolean", "String")%
                case NPTypeCode.#1:
                    return #1.MinValue;
                %
#else
                case NPTypeCode.SByte:
                    return SByte.MinValue;
                case NPTypeCode.Byte:
                    return Byte.MinValue;
                case NPTypeCode.Int16:
                    return Int16.MinValue;
                case NPTypeCode.UInt16:
                    return UInt16.MinValue;
                case NPTypeCode.Int32:
                    return Int32.MinValue;
                case NPTypeCode.UInt32:
                    return UInt32.MinValue;
                case NPTypeCode.Int64:
                    return Int64.MinValue;
                case NPTypeCode.UInt64:
                    return UInt64.MinValue;
                case NPTypeCode.Char:
                    return Char.MinValue;
                case NPTypeCode.Double:
                    return Double.MinValue;
                case NPTypeCode.Single:
                    return Single.MinValue;
                case NPTypeCode.Decimal:
                    return Decimal.MinValue;
#endif
                default:
                    throw new ArgumentOutOfRangeException(nameof(typeCode), typeCode, null);
            }
        }
    }
}
