namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Minimum element counts for SIMD to be beneficial.
    /// Below these thresholds, the overhead of SIMD setup may exceed the benefits.
    /// Based on Vector256 (32 bytes) width.
    /// </summary>
    public static class SimdThresholds
    {
        /// <summary>Minimum elements for byte (32 per Vector256).</summary>
        public const int Byte = 64;

        /// <summary>Minimum elements for Int16/UInt16 (16 per Vector256).</summary>
        public const int Int16 = 64;

        /// <summary>Minimum elements for Int32/UInt32/Single (8 per Vector256).</summary>
        public const int Int32 = 96;

        /// <summary>Minimum elements for Int64/UInt64/Double (4 per Vector256).</summary>
        public const int Int64 = 256;

        /// <summary>Minimum elements for Single (8 per Vector256).</summary>
        public const int Single = 96;

        /// <summary>Minimum elements for Double (4 per Vector256) - conservative.</summary>
        public const int Double = 512;

        /// <summary>
        /// Size above which memory bandwidth dominates and SIMD speedup diminishes.
        /// At very large sizes, we're limited by memory bandwidth, not compute.
        /// </summary>
        public const int MemoryBound = 10_000_000;

        /// <summary>
        /// Get the minimum threshold for a given NPTypeCode.
        /// </summary>
        public static int GetThreshold(NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.Boolean => Byte,
                NPTypeCode.Byte => Byte,
                NPTypeCode.Int16 => Int16,
                NPTypeCode.UInt16 => Int16,
                NPTypeCode.Int32 => Int32,
                NPTypeCode.UInt32 => Int32,
                NPTypeCode.Int64 => Int64,
                NPTypeCode.UInt64 => Int64,
                NPTypeCode.Single => Single,
                NPTypeCode.Double => Double,
                NPTypeCode.Char => Int16,
                NPTypeCode.Decimal => Int64, // No SIMD for Decimal, use high threshold
                _ => Int32
            };
        }

        /// <summary>
        /// Returns true if the array size is above the SIMD threshold for the given type.
        /// </summary>
        public static bool ShouldUseSIMD(NPTypeCode typeCode, int size)
        {
            return size >= GetThreshold(typeCode);
        }
    }
}
