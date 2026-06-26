namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Central registry for every runtime-generated kernel cache. Each cache exposes a
    ///     public read-only <c>*Count</c> (live cache size) and an <c>internal</c> reset.
    /// </summary>
    /// <remarks>
    ///     Counts are public observability: callers may read them but cannot mutate the caches,
    ///     which fill automatically as kernels are JIT-emitted on first use and reused thereafter.
    ///     Resetting a cache (<c>Clear*</c>) is internal, test-only — clearing forces recompilation.
    ///     The reflection <see cref="VectorMethodCache"/>/<see cref="ScalarMethodCache"/> method
    ///     caches are not generated delegates and are intentionally excluded.
    /// </remarks>
    public static class GeneratedDelegates
    {
        // ---- Per-cache live size (public) ----

        // DirectILKernelGenerator.Argwhere.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._argwhereCount</c>.</summary>
        public static int ArgwhereCount => DirectILKernelGenerator._argwhereCount.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._argwhereFlat</c>.</summary>
        public static int ArgwhereFlatCount => DirectILKernelGenerator._argwhereFlat.Count;
        // DirectILKernelGenerator.Binary.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._contiguousKernelCache</c>.</summary>
        public static int ContiguousKernelCount => DirectILKernelGenerator._contiguousKernelCache.Count;
        // DirectILKernelGenerator.Cast.Masked.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._maskedCastCache</c>.</summary>
        public static int MaskedCastCount => DirectILKernelGenerator._maskedCastCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._maskedCastUnsupported</c>.</summary>
        public static int MaskedCastUnsupportedCount => DirectILKernelGenerator._maskedCastUnsupported.Count;
        // DirectILKernelGenerator.Cast.Scalar.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._innerCastCache</c>.</summary>
        public static int InnerCastCount => DirectILKernelGenerator._innerCastCache.Count;
        // DirectILKernelGenerator.Cast.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._castCache</c>.</summary>
        public static int CastCount => DirectILKernelGenerator._castCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._castUnsupported</c>.</summary>
        public static int CastUnsupportedCount => DirectILKernelGenerator._castUnsupported.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._stridedCastCache</c>.</summary>
        public static int StridedCastCount => DirectILKernelGenerator._stridedCastCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._stridedCastUnsupported</c>.</summary>
        public static int StridedCastUnsupportedCount => DirectILKernelGenerator._stridedCastUnsupported.Count;
        // DirectILKernelGenerator.Clip.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._clipKernelCache</c>.</summary>
        public static int ClipKernelCount => DirectILKernelGenerator._clipKernelCache.Count;
        // DirectILKernelGenerator.Comparison.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._comparisonCache</c>.</summary>
        public static int ComparisonCount => DirectILKernelGenerator._comparisonCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._comparisonScalarCache</c>.</summary>
        public static int ComparisonScalarCount => DirectILKernelGenerator._comparisonScalarCache.Count;
        // DirectILKernelGenerator.Copy.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._copyKernelCache</c>.</summary>
        public static int CopyKernelCount => DirectILKernelGenerator._copyKernelCache.Count;
        // DirectILKernelGenerator.Filter.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._filterAxis</c>.</summary>
        public static int FilterAxisCount => DirectILKernelGenerator._filterAxis.Count;
        // DirectILKernelGenerator.InnerLoop.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._innerLoopCache</c>.</summary>
        public static int InnerLoopCount => DirectILKernelGenerator._innerLoopCache.Count;
        // DirectILKernelGenerator.MatMul.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._matmulKernelCache</c>.</summary>
        public static int MatmulKernelCount => DirectILKernelGenerator._matmulKernelCache.Count;
        // DirectILKernelGenerator.MixedType.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._mixedTypeCache</c>.</summary>
        public static int MixedTypeCount => DirectILKernelGenerator._mixedTypeCache.Count;
        // DirectILKernelGenerator.Quantile.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._quantileKernelCache</c>.</summary>
        public static int QuantileKernelCount => DirectILKernelGenerator._quantileKernelCache.Count;
        // DirectILKernelGenerator.Reduction.Axis.Boolean.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._boolAxisReductionCache</c>.</summary>
        public static int BoolAxisReductionCount => DirectILKernelGenerator._boolAxisReductionCache.Count;
        // DirectILKernelGenerator.Reduction.Axis.NaN.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._nanAxisReductionCache</c>.</summary>
        public static int NanAxisReductionCount => DirectILKernelGenerator._nanAxisReductionCache.Count;
        // DirectILKernelGenerator.Reduction.Axis.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._axisReductionCache</c>.</summary>
        public static int AxisReductionCount => DirectILKernelGenerator._axisReductionCache.Count;
        // DirectILKernelGenerator.Reduction.NaN.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._nanElementReductionCache</c>.</summary>
        public static int NanElementReductionCount => DirectILKernelGenerator._nanElementReductionCache.Count;
        // DirectILKernelGenerator.Reduction.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._elementReductionCache</c>.</summary>
        public static int ElementReductionCount => DirectILKernelGenerator._elementReductionCache.Count;
        // DirectILKernelGenerator.Repeat.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._repeatBroadcastCache</c>.</summary>
        public static int RepeatBroadcastCount => DirectILKernelGenerator._repeatBroadcastCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._repeatPerJCache</c>.</summary>
        public static int RepeatPerJCount => DirectILKernelGenerator._repeatPerJCache.Count;
        // DirectILKernelGenerator.Scalar.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._binaryScalarCache</c>.</summary>
        public static int BinaryScalarCount => DirectILKernelGenerator._binaryScalarCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._unaryScalarCache</c>.</summary>
        public static int UnaryScalarCount => DirectILKernelGenerator._unaryScalarCache.Count;
        // DirectILKernelGenerator.Scan.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._axisScanCache</c>.</summary>
        public static int AxisScanCount => DirectILKernelGenerator._axisScanCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._scanCache</c>.</summary>
        public static int ScanCount => DirectILKernelGenerator._scanCache.Count;
        // DirectILKernelGenerator.Search.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._searchCache</c>.</summary>
        public static int SearchCount => DirectILKernelGenerator._searchCache.Count;
        // DirectILKernelGenerator.Shift.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._shiftKernelCache</c>.</summary>
        public static int ShiftKernelCount => DirectILKernelGenerator._shiftKernelCache.Count;
        // DirectILKernelGenerator.StorageAlias.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._storageAliasFieldCopiers</c>.</summary>
        public static int StorageAliasFieldCopiersCount => DirectILKernelGenerator._storageAliasFieldCopiers.Count;
        // DirectILKernelGenerator.Trace.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._trace</c>.</summary>
        public static int TraceCount => DirectILKernelGenerator._trace.Count;
        // DirectILKernelGenerator.Unary.Strided.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._stridedUnaryCache</c>.</summary>
        public static int StridedUnaryCount => DirectILKernelGenerator._stridedUnaryCache.Count;
        // DirectILKernelGenerator.Unary.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._unaryCache</c>.</summary>
        public static int UnaryCount => DirectILKernelGenerator._unaryCache.Count;
        // DirectILKernelGenerator.WeightedSum.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._weightedSumCache</c>.</summary>
        public static int WeightedSumCount => DirectILKernelGenerator._weightedSumCache.Count;
        // DirectILKernelGenerator.Where.Scalar.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._whereScalarXCache</c>.</summary>
        public static int WhereScalarXCount => DirectILKernelGenerator._whereScalarXCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._whereScalarXYCache</c>.</summary>
        public static int WhereScalarXYCount => DirectILKernelGenerator._whereScalarXYCache.Count;
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._whereScalarYCache</c>.</summary>
        public static int WhereScalarYCount => DirectILKernelGenerator._whereScalarYCache.Count;
        // DirectILKernelGenerator.Where.cs
        /// <summary>Cached kernels in <c>DirectILKernelGenerator._whereKernelCache</c>.</summary>
        public static int WhereKernelCount => DirectILKernelGenerator._whereKernelCache.Count;
        // ILKernelGenerator.Reduction.Pairwise.cs
        /// <summary>Cached kernels in <c>ILKernelGenerator._pwFolds</c>.</summary>
        public static int PwFoldsCount => ILKernelGenerator._pwFolds.Count;
        // ILKernelGenerator.Reduction.cs
        /// <summary>Cached kernels in <c>ILKernelGenerator._reduceCache</c>.</summary>
        public static int ReduceCount => ILKernelGenerator._reduceCache.Count;
        // ILKernelGenerator.Scan.cs
        /// <summary>Cached kernels in <c>ILKernelGenerator._cumSumCache</c>.</summary>
        public static int CumSumCount => ILKernelGenerator._cumSumCache.Count;
        // ILKernelGenerator.Where.cs
        /// <summary>Cached kernels in <c>ILKernelGenerator._whereInnerCache</c>.</summary>
        public static int WhereInnerCount => ILKernelGenerator._whereInnerCache.Count;

        /// <summary>Total generated kernels across every cache above.</summary>
        public static int TotalCount =>
            ArgwhereCount + ArgwhereFlatCount + ContiguousKernelCount + MaskedCastCount +
            MaskedCastUnsupportedCount + InnerCastCount + CastCount + CastUnsupportedCount +
            StridedCastCount + StridedCastUnsupportedCount + ClipKernelCount + ComparisonCount +
            ComparisonScalarCount + CopyKernelCount + FilterAxisCount + InnerLoopCount +
            MatmulKernelCount + MixedTypeCount + QuantileKernelCount + BoolAxisReductionCount +
            NanAxisReductionCount + AxisReductionCount + NanElementReductionCount +
            ElementReductionCount + RepeatBroadcastCount + RepeatPerJCount + BinaryScalarCount +
            UnaryScalarCount + AxisScanCount + ScanCount + SearchCount + ShiftKernelCount +
            StorageAliasFieldCopiersCount + TraceCount + StridedUnaryCount + UnaryCount +
            WeightedSumCount + WhereScalarXCount + WhereScalarXYCount + WhereScalarYCount +
            WhereKernelCount + PwFoldsCount + ReduceCount + CumSumCount + WhereInnerCount;

        // ---- Per-cache reset (internal, test-only) ----

        internal static void ClearArgwhereCount() => DirectILKernelGenerator._argwhereCount.Clear();
        internal static void ClearArgwhereFlat() => DirectILKernelGenerator._argwhereFlat.Clear();
        internal static void ClearContiguousKernel() => DirectILKernelGenerator._contiguousKernelCache.Clear();
        internal static void ClearMaskedCast() => DirectILKernelGenerator._maskedCastCache.Clear();
        internal static void ClearMaskedCastUnsupported() => DirectILKernelGenerator._maskedCastUnsupported.Clear();
        internal static void ClearInnerCast() => DirectILKernelGenerator._innerCastCache.Clear();
        internal static void ClearCast() => DirectILKernelGenerator._castCache.Clear();
        internal static void ClearCastUnsupported() => DirectILKernelGenerator._castUnsupported.Clear();
        internal static void ClearStridedCast() => DirectILKernelGenerator._stridedCastCache.Clear();
        internal static void ClearStridedCastUnsupported() => DirectILKernelGenerator._stridedCastUnsupported.Clear();
        internal static void ClearClipKernel() => DirectILKernelGenerator._clipKernelCache.Clear();
        internal static void ClearComparison() => DirectILKernelGenerator._comparisonCache.Clear();
        internal static void ClearComparisonScalar() => DirectILKernelGenerator._comparisonScalarCache.Clear();
        internal static void ClearCopyKernel() => DirectILKernelGenerator._copyKernelCache.Clear();
        internal static void ClearFilterAxis() => DirectILKernelGenerator._filterAxis.Clear();
        internal static void ClearInnerLoop() => DirectILKernelGenerator._innerLoopCache.Clear();
        internal static void ClearMatmulKernel() => DirectILKernelGenerator._matmulKernelCache.Clear();
        internal static void ClearMixedType() => DirectILKernelGenerator._mixedTypeCache.Clear();
        internal static void ClearQuantileKernel() => DirectILKernelGenerator._quantileKernelCache.Clear();
        internal static void ClearBoolAxisReduction() => DirectILKernelGenerator._boolAxisReductionCache.Clear();
        internal static void ClearNanAxisReduction() => DirectILKernelGenerator._nanAxisReductionCache.Clear();
        internal static void ClearAxisReduction() => DirectILKernelGenerator._axisReductionCache.Clear();
        internal static void ClearNanElementReduction() => DirectILKernelGenerator._nanElementReductionCache.Clear();
        internal static void ClearElementReduction() => DirectILKernelGenerator._elementReductionCache.Clear();
        internal static void ClearRepeatBroadcast() => DirectILKernelGenerator._repeatBroadcastCache.Clear();
        internal static void ClearRepeatPerJ() => DirectILKernelGenerator._repeatPerJCache.Clear();
        internal static void ClearBinaryScalar() => DirectILKernelGenerator._binaryScalarCache.Clear();
        internal static void ClearUnaryScalar() => DirectILKernelGenerator._unaryScalarCache.Clear();
        internal static void ClearAxisScan() => DirectILKernelGenerator._axisScanCache.Clear();
        internal static void ClearScan() => DirectILKernelGenerator._scanCache.Clear();
        internal static void ClearSearch() => DirectILKernelGenerator._searchCache.Clear();
        internal static void ClearShiftKernel() => DirectILKernelGenerator._shiftKernelCache.Clear();
        internal static void ClearStorageAliasFieldCopiers() => DirectILKernelGenerator._storageAliasFieldCopiers.Clear();
        internal static void ClearTrace() => DirectILKernelGenerator._trace.Clear();
        internal static void ClearStridedUnary() => DirectILKernelGenerator._stridedUnaryCache.Clear();
        internal static void ClearUnary() => DirectILKernelGenerator._unaryCache.Clear();
        internal static void ClearWeightedSum() => DirectILKernelGenerator._weightedSumCache.Clear();
        internal static void ClearWhereScalarX() => DirectILKernelGenerator._whereScalarXCache.Clear();
        internal static void ClearWhereScalarXY() => DirectILKernelGenerator._whereScalarXYCache.Clear();
        internal static void ClearWhereScalarY() => DirectILKernelGenerator._whereScalarYCache.Clear();
        internal static void ClearWhereKernel() => DirectILKernelGenerator._whereKernelCache.Clear();
        internal static void ClearPwFolds() => ILKernelGenerator._pwFolds.Clear();
        internal static void ClearReduce() => ILKernelGenerator._reduceCache.Clear();
        internal static void ClearCumSum() => ILKernelGenerator._cumSumCache.Clear();
        internal static void ClearWhereInner() => ILKernelGenerator._whereInnerCache.Clear();

        /// <summary>Reset every generated-kernel cache (test-only).</summary>
        internal static void ClearAll()
        {
            ClearArgwhereCount();
            ClearArgwhereFlat();
            ClearContiguousKernel();
            ClearMaskedCast();
            ClearMaskedCastUnsupported();
            ClearInnerCast();
            ClearCast();
            ClearCastUnsupported();
            ClearStridedCast();
            ClearStridedCastUnsupported();
            ClearClipKernel();
            ClearComparison();
            ClearComparisonScalar();
            ClearCopyKernel();
            ClearFilterAxis();
            ClearInnerLoop();
            ClearMatmulKernel();
            ClearMixedType();
            ClearQuantileKernel();
            ClearBoolAxisReduction();
            ClearNanAxisReduction();
            ClearAxisReduction();
            ClearNanElementReduction();
            ClearElementReduction();
            ClearRepeatBroadcast();
            ClearRepeatPerJ();
            ClearBinaryScalar();
            ClearUnaryScalar();
            ClearAxisScan();
            ClearScan();
            ClearSearch();
            ClearShiftKernel();
            ClearStorageAliasFieldCopiers();
            ClearTrace();
            ClearStridedUnary();
            ClearUnary();
            ClearWeightedSum();
            ClearWhereScalarX();
            ClearWhereScalarXY();
            ClearWhereScalarY();
            ClearWhereKernel();
            ClearPwFolds();
            ClearReduce();
            ClearCumSum();
            ClearWhereInner();
        }
    }
}
