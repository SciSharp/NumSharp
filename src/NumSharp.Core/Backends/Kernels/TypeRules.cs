using System;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Shared type rules for kernel providers.
    /// </summary>
    public static class TypeRules
    {
        /// <summary>Get size in bytes for NPTypeCode.</summary>
        public static int GetTypeSize(NPTypeCode type) => type switch
        {
            NPTypeCode.Boolean => 1,
            NPTypeCode.Byte => 1,
            NPTypeCode.Int16 => 2,
            NPTypeCode.UInt16 => 2,
            NPTypeCode.Char => 2,
            NPTypeCode.Int32 => 4,
            NPTypeCode.UInt32 => 4,
            NPTypeCode.Single => 4,
            NPTypeCode.Int64 => 8,
            NPTypeCode.UInt64 => 8,
            NPTypeCode.Double => 8,
            NPTypeCode.Decimal => 16,
            _ => throw new NotSupportedException($"Type {type} not supported")
        };

        /// <summary>Get CLR Type for NPTypeCode.</summary>
        public static Type GetClrType(NPTypeCode type) => type switch
        {
            NPTypeCode.Boolean => typeof(bool),
            NPTypeCode.Byte => typeof(byte),
            NPTypeCode.Int16 => typeof(short),
            NPTypeCode.UInt16 => typeof(ushort),
            NPTypeCode.Char => typeof(char),
            NPTypeCode.Int32 => typeof(int),
            NPTypeCode.UInt32 => typeof(uint),
            NPTypeCode.Single => typeof(float),
            NPTypeCode.Int64 => typeof(long),
            NPTypeCode.UInt64 => typeof(ulong),
            NPTypeCode.Double => typeof(double),
            NPTypeCode.Decimal => typeof(decimal),
            _ => throw new NotSupportedException($"Type {type} not supported")
        };

        /// <summary>
        /// Get accumulating type for reductions (NEP50 alignment).
        /// int32/int16/byte/bool → int64, uint32/uint16 → uint64, floats preserve type.
        /// NumPy: Boolean arrays are treated as integers for accumulation (True=1, False=0).
        /// </summary>
        public static NPTypeCode GetAccumulatingType(NPTypeCode type) => type switch
        {
            NPTypeCode.Int32 or NPTypeCode.Int16 or NPTypeCode.Byte or NPTypeCode.Boolean => NPTypeCode.Int64,
            NPTypeCode.UInt32 or NPTypeCode.UInt16 => NPTypeCode.UInt64,
            _ => type  // Float/Double/Decimal/Int64/UInt64 preserve type
        };

        /// <summary>Check if type is unsigned integer.</summary>
        public static bool IsUnsigned(NPTypeCode type) =>
            type is NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64;

        /// <summary>Check if type is signed integer.</summary>
        public static bool IsSigned(NPTypeCode type) =>
            type is NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64;

        /// <summary>Check if type is floating point.</summary>
        public static bool IsFloatingPoint(NPTypeCode type) =>
            type is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal;

        /// <summary>Check if type is integer (signed or unsigned).</summary>
        public static bool IsInteger(NPTypeCode type) =>
            IsSigned(type) || IsUnsigned(type);

        /// <summary>Get elements per vector for type at given vector width.</summary>
        public static int GetVectorCount(NPTypeCode type, int vectorBits) =>
            vectorBits == 0 ? 1 : (vectorBits / 8) / GetTypeSize(type);

        /// <summary>Check if type can use SIMD operations.</summary>
        public static bool CanUseSimd(NPTypeCode type) =>
            type is not (NPTypeCode.Boolean or NPTypeCode.Char or NPTypeCode.Decimal);
    }
}
