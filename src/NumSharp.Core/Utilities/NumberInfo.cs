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
                case NPTypeCode.Byte:
                    return Byte.MaxValue;
                case NPTypeCode.Int32:
                    return Int32.MaxValue;
                case NPTypeCode.Int64:
                    return Int64.MaxValue;
                case NPTypeCode.Single:
                    return Single.MaxValue;
                case NPTypeCode.Double:
                    return Double.MaxValue;
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
#if _REGEN1
	            %foreach except(supported_primitives, "Boolean", "String")%
                case NPTypeCode.#1:
                    return #1.MinValue;
                %
#else
                case NPTypeCode.Byte:
                    return Byte.MinValue;
                case NPTypeCode.Int32:
                    return Int32.MinValue;
                case NPTypeCode.Int64:
                    return Int64.MinValue;
                case NPTypeCode.Single:
                    return Single.MinValue;
                case NPTypeCode.Double:
                    return Double.MinValue;
#endif
                default:
                    throw new ArgumentOutOfRangeException(nameof(typeCode), typeCode, null);
            }
        }
    }
}
