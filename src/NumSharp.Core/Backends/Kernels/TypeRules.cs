using System;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Shared type rules for kernel providers.
    /// </summary>
    /// <remarks>
    /// This class delegates to NPTypeCode extension methods for consistency.
    /// Prefer using NPTypeCode extensions directly (e.g., typeCode.SizeOf(), typeCode.AsType()).
    /// </remarks>
    public static class TypeRules
    {
        /// <summary>Get size in bytes for NPTypeCode (CLR memory size).</summary>
        /// <remarks>
        /// Returns actual CLR memory size for pointer arithmetic in IL generation.
        /// Note: This differs from <see cref="NPTypeCodeExtensions.SizeOf"/> which uses numpy-compatible sizes.
        /// </remarks>
        public static int GetTypeSize(NPTypeCode type) => type switch
        {
            NPTypeCode.Boolean => 1,
            NPTypeCode.Byte => 1,
            NPTypeCode.Int16 => 2,
            NPTypeCode.UInt16 => 2,
            NPTypeCode.Char => 2,  // CLR char is 2 bytes (UTF-16)
            NPTypeCode.Int32 => 4,
            NPTypeCode.UInt32 => 4,
            NPTypeCode.Single => 4,
            NPTypeCode.Int64 => 8,
            NPTypeCode.UInt64 => 8,
            NPTypeCode.Double => 8,
            NPTypeCode.Decimal => 16,  // CLR decimal is 16 bytes
            _ => throw new NotSupportedException($"Type {type} not supported")
        };

        /// <summary>Get CLR Type for NPTypeCode.</summary>
        /// <remarks>Delegates to <see cref="NPTypeCodeExtensions.AsType"/>.</remarks>
        public static Type GetClrType(NPTypeCode type) => type.AsType();

        /// <summary>
        /// Get accumulating type for reductions (NEP50 alignment).
        /// </summary>
        /// <remarks>Delegates to <see cref="NPTypeCodeExtensions.GetAccumulatingType"/>.</remarks>
        public static NPTypeCode GetAccumulatingType(NPTypeCode type) => type.GetAccumulatingType();

        /// <summary>Check if type is unsigned integer.</summary>
        /// <remarks>
        /// Note: This returns true only for unsigned integer types (Byte, UInt16, UInt32, UInt64).
        /// For a version that also includes Boolean and Char, use <see cref="NPTypeCodeExtensions.IsUnsigned"/>.
        /// </remarks>
        public static bool IsUnsigned(NPTypeCode type) =>
            type is NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64;

        /// <summary>Check if type is signed integer.</summary>
        /// <remarks>
        /// Note: This returns true only for signed integer types (Int16, Int32, Int64).
        /// For a version that also includes floats, use <see cref="NPTypeCodeExtensions.IsSigned"/>.
        /// </remarks>
        public static bool IsSigned(NPTypeCode type) =>
            type is NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64;

        /// <summary>Check if type is floating point.</summary>
        /// <remarks>Delegates to <see cref="NPTypeCodeExtensions.IsFloatingPoint"/>.</remarks>
        public static bool IsFloatingPoint(NPTypeCode type) => type.IsFloatingPoint();

        /// <summary>Check if type is integer (signed or unsigned).</summary>
        /// <remarks>Delegates to <see cref="NPTypeCodeExtensions.IsInteger"/>.</remarks>
        public static bool IsInteger(NPTypeCode type) => type.IsInteger();

        /// <summary>Get elements per vector for type at given vector width.</summary>
        public static int GetVectorCount(NPTypeCode type, int vectorBits) =>
            vectorBits == 0 ? 1 : (vectorBits / 8) / GetTypeSize(type);

        /// <summary>Check if type can use SIMD operations.</summary>
        /// <remarks>Delegates to <see cref="NPTypeCodeExtensions.IsSimdCapable"/>.</remarks>
        public static bool CanUseSimd(NPTypeCode type) => type.IsSimdCapable();
    }
}
