namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Centralized, read-only view over how many runtime-generated kernel delegates
    ///     each cache currently holds.
    /// </summary>
    /// <remarks>
    ///     Every count reflects the <b>live</b> size of the corresponding kernel cache.
    ///     Callers may observe these numbers but cannot mutate them: the values change
    ///     only as a side effect of internal JIT-emission/caching as kernels are compiled
    ///     on first use and reused thereafter. This is the single public surface for kernel
    ///     cache observability — the individual caches and their counts are internal.
    /// </remarks>
    public static class GeneratedDelegates
    {
        // ---- Elementwise (whole-array, DirectILKernelGenerator) ----

        /// <summary>Contiguous binary kernels (Add/Sub/Mul/… on same-type operands).</summary>
        public static int BinaryCount => DirectILKernelGenerator._contiguousKernelCache.Count;

        /// <summary>Mixed-type binary kernels (operands of differing dtype).</summary>
        public static int MixedTypeCount => DirectILKernelGenerator._mixedTypeCache.Count;

        /// <summary>Contiguous unary kernels (Negate/Abs/Sqrt/trig/…).</summary>
        public static int UnaryCount => DirectILKernelGenerator._unaryCache.Count;

        /// <summary>Strided unary kernels.</summary>
        public static int StridedUnaryCount => DirectILKernelGenerator._stridedUnaryCache.Count;

        /// <summary>Scalar (non-SIMD, e.g. Decimal) unary kernels.</summary>
        public static int UnaryScalarCount => DirectILKernelGenerator._unaryScalarCache.Count;

        /// <summary>Scalar (non-SIMD) binary kernels.</summary>
        public static int BinaryScalarCount => DirectILKernelGenerator._binaryScalarCache.Count;

        // ---- Comparison ----

        /// <summary>Array-vs-array comparison kernels.</summary>
        public static int ComparisonCount => DirectILKernelGenerator._comparisonCache.Count;

        /// <summary>Array-vs-scalar comparison kernels.</summary>
        public static int ComparisonScalarCount => DirectILKernelGenerator._comparisonScalarCache.Count;

        // ---- Cast ----

        /// <summary>Contiguous cast kernels.</summary>
        public static int CastCount => DirectILKernelGenerator._castCache.Count;

        /// <summary>Strided cast kernels.</summary>
        public static int StridedCastCount => DirectILKernelGenerator._stridedCastCache.Count;

        /// <summary>Masked cast kernels.</summary>
        public static int MaskedCastCount => DirectILKernelGenerator._maskedCastCache.Count;

        /// <summary>Scalar inner-cast kernels.</summary>
        public static int InnerCastCount => DirectILKernelGenerator._innerCastCache.Count;

        // ---- Reduction ----

        /// <summary>Flat element reduction kernels (Sum/Prod/Min/Max/…).</summary>
        public static int ElementReductionCount => DirectILKernelGenerator._elementReductionCache.Count;

        /// <summary>Flat NaN-aware element reduction kernels.</summary>
        public static int NanElementReductionCount => DirectILKernelGenerator._nanElementReductionCache.Count;

        /// <summary>Axis reduction kernels.</summary>
        public static int AxisReductionCount => DirectILKernelGenerator._axisReductionCache.Count;

        /// <summary>Boolean axis reduction kernels (All/Any).</summary>
        public static int BooleanAxisReductionCount => DirectILKernelGenerator._boolAxisReductionCache.Count;

        /// <summary>NaN-aware axis reduction kernels.</summary>
        public static int NanAxisReductionCount => DirectILKernelGenerator._nanAxisReductionCache.Count;

        // ---- Scan / Search ----

        /// <summary>Flat scan kernels (CumSum/CumProd).</summary>
        public static int ScanCount => DirectILKernelGenerator._scanCache.Count;

        /// <summary>Axis scan kernels.</summary>
        public static int AxisScanCount => DirectILKernelGenerator._axisScanCache.Count;

        /// <summary>searchsorted kernels.</summary>
        public static int SearchCount => DirectILKernelGenerator._searchCache.Count;

        // ---- NpyIter-driven per-chunk kernels ----

        /// <summary>Fused-expression / pairwise inner-loop kernels (np.evaluate, flat reductions).</summary>
        public static int InnerLoopCount => DirectILKernelGenerator._innerLoopCache.Count;

        /// <summary>Sum of every generated-kernel cache tracked above.</summary>
        public static int TotalCount =>
            BinaryCount + MixedTypeCount + UnaryCount + StridedUnaryCount
            + UnaryScalarCount + BinaryScalarCount
            + ComparisonCount + ComparisonScalarCount
            + CastCount + StridedCastCount + MaskedCastCount + InnerCastCount
            + ElementReductionCount + NanElementReductionCount + AxisReductionCount
            + BooleanAxisReductionCount + NanAxisReductionCount
            + ScanCount + AxisScanCount + SearchCount + InnerLoopCount;
    }
}
