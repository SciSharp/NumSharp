using System;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Reduction operation types supported by the IL kernel infrastructure.
    /// </summary>
    public enum ReductionOp
    {
        /// <summary>Sum of elements (add reduction)</summary>
        Sum,
        /// <summary>Product of elements (multiply reduction)</summary>
        Prod,
        /// <summary>Maximum element</summary>
        Max,
        /// <summary>Minimum element</summary>
        Min,
        /// <summary>Index of maximum element (returns int)</summary>
        ArgMax,
        /// <summary>Index of minimum element (returns int)</summary>
        ArgMin,
        /// <summary>Mean = Sum / count</summary>
        Mean,
        /// <summary>Cumulative sum (running total)</summary>
        CumSum
    }

    /// <summary>
    /// Execution path for reduction operations.
    /// </summary>
    public enum ReductionPath
    {
        /// <summary>Reduce all elements to a single scalar value (no axis specified)</summary>
        ElementWise,
        /// <summary>Reduce along a specific axis, producing smaller-dimensional result</summary>
        AxisReduction,
        /// <summary>Cumulative reduction - same shape output, running accumulation</summary>
        Cumulative
    }

    /// <summary>
    /// Cache key for element-wise (full array) reduction kernels.
    /// Reduces all elements to a single scalar value.
    /// </summary>
    /// <remarks>
    /// Supports up to 144 unique kernels: 12 types × 8 operations × 1 path
    /// </remarks>
    public readonly record struct ElementReductionKernelKey(
        NPTypeCode InputType,
        NPTypeCode AccumulatorType,
        ReductionOp Op,
        bool IsContiguous
    )
    {
        /// <summary>
        /// Returns true if input and accumulator types are the same.
        /// </summary>
        public bool IsSameType => InputType == AccumulatorType;

        /// <summary>
        /// Result type depends on operation:
        /// - ArgMax/ArgMin: always Int32
        /// - Mean: always Double (or accumulator type if specified)
        /// - Others: same as accumulator type
        /// </summary>
        public NPTypeCode ResultType => Op switch
        {
            ReductionOp.ArgMax => NPTypeCode.Int32,
            ReductionOp.ArgMin => NPTypeCode.Int32,
            _ => AccumulatorType
        };

        public override string ToString() =>
            $"Elem_{Op}_{InputType}_{AccumulatorType}_{(IsContiguous ? "Contig" : "Strided")}";
    }

    /// <summary>
    /// Cache key for axis reduction kernels.
    /// Reduces along a specific axis, producing an array with one fewer dimension.
    /// </summary>
    /// <remarks>
    /// These kernels handle the outer loop over non-reduced dimensions
    /// and inner reduction along the specified axis.
    /// </remarks>
    public readonly record struct AxisReductionKernelKey(
        NPTypeCode InputType,
        NPTypeCode AccumulatorType,
        ReductionOp Op,
        bool InnerAxisContiguous
    )
    {
        /// <summary>
        /// Returns true if input and accumulator types are the same.
        /// </summary>
        public bool IsSameType => InputType == AccumulatorType;

        /// <summary>
        /// Result type depends on operation.
        /// </summary>
        public NPTypeCode ResultType => Op switch
        {
            ReductionOp.ArgMax => NPTypeCode.Int32,
            ReductionOp.ArgMin => NPTypeCode.Int32,
            _ => AccumulatorType
        };

        public override string ToString() =>
            $"Axis_{Op}_{InputType}_{AccumulatorType}_{(InnerAxisContiguous ? "InnerContig" : "Strided")}";
    }

    /// <summary>
    /// Cache key for cumulative reduction kernels (cumsum, etc.).
    /// Output has same shape as input, each element is accumulation of elements before it.
    /// </summary>
    public readonly record struct CumulativeKernelKey(
        NPTypeCode InputType,
        NPTypeCode OutputType,
        ReductionOp Op,
        bool IsContiguous
    )
    {
        public bool IsSameType => InputType == OutputType;

        public override string ToString() =>
            $"Cum_{Op}_{InputType}_{OutputType}_{(IsContiguous ? "Contig" : "Strided")}";
    }

    /// <summary>
    /// Delegate for element-wise reduction kernels.
    /// Reduces all elements of an array to a single value.
    /// </summary>
    /// <param name="input">Pointer to input data</param>
    /// <param name="strides">Input strides (element units, not bytes)</param>
    /// <param name="shape">Input shape dimensions</param>
    /// <param name="ndim">Number of dimensions</param>
    /// <param name="totalSize">Total number of elements</param>
    /// <returns>The reduced value (boxed)</returns>
    /// <remarks>
    /// Returns object to handle different accumulator types without generic delegates.
    /// The caller unboxes based on the kernel key's AccumulatorType.
    /// For ArgMax/ArgMin, returns the index as int.
    /// </remarks>
    public unsafe delegate object ElementReductionKernel(
        void* input,
        int* strides,
        int* shape,
        int ndim,
        int totalSize
    );

    /// <summary>
    /// Delegate for typed element-wise reduction kernels.
    /// Returns the reduced value directly without boxing.
    /// </summary>
    /// <typeparam name="TResult">Accumulator/result type</typeparam>
    public unsafe delegate TResult TypedElementReductionKernel<TResult>(
        void* input,
        int* strides,
        int* shape,
        int ndim,
        int totalSize
    ) where TResult : unmanaged;

    /// <summary>
    /// Delegate for axis reduction kernels.
    /// Reduces along a specific axis, writing to output array.
    /// </summary>
    /// <param name="input">Pointer to input data</param>
    /// <param name="output">Pointer to output data</param>
    /// <param name="inputStrides">Input strides (element units)</param>
    /// <param name="inputShape">Input shape dimensions</param>
    /// <param name="outputStrides">Output strides (element units)</param>
    /// <param name="axis">Axis to reduce along</param>
    /// <param name="axisSize">Size of the axis being reduced</param>
    /// <param name="ndim">Number of input dimensions</param>
    /// <param name="outputSize">Total number of output elements</param>
    public unsafe delegate void AxisReductionKernel(
        void* input,
        void* output,
        int* inputStrides,
        int* inputShape,
        int* outputStrides,
        int axis,
        int axisSize,
        int ndim,
        int outputSize
    );

    /// <summary>
    /// Delegate for cumulative reduction kernels (cumsum, etc.).
    /// Output has same shape as input.
    /// </summary>
    /// <param name="input">Pointer to input data</param>
    /// <param name="output">Pointer to output data</param>
    /// <param name="strides">Input strides (element units)</param>
    /// <param name="shape">Shape dimensions</param>
    /// <param name="ndim">Number of dimensions</param>
    /// <param name="totalSize">Total number of elements</param>
    public unsafe delegate void CumulativeKernel(
        void* input,
        void* output,
        int* strides,
        int* shape,
        int ndim,
        int totalSize
    );

    /// <summary>
    /// Delegate for cumulative axis reduction kernels.
    /// Computes running accumulation along a specific axis.
    /// </summary>
    /// <param name="input">Pointer to input data</param>
    /// <param name="output">Pointer to output data</param>
    /// <param name="inputStrides">Input strides (element units)</param>
    /// <param name="shape">Shape dimensions</param>
    /// <param name="axis">Axis to accumulate along</param>
    /// <param name="ndim">Number of dimensions</param>
    /// <param name="totalSize">Total number of elements</param>
    public unsafe delegate void CumulativeAxisKernel(
        void* input,
        void* output,
        int* inputStrides,
        int* shape,
        int axis,
        int ndim,
        int totalSize
    );

    /// <summary>
    /// Extension methods for ReductionOp.
    /// </summary>
    public static class ReductionOpExtensions
    {
        /// <summary>
        /// Get the identity element for this reduction operation.
        /// </summary>
        public static object GetIdentity(this ReductionOp op, NPTypeCode type)
        {
            return op switch
            {
                ReductionOp.Sum => type.GetDefaultValue(),
                ReductionOp.Prod => type.GetOneValue(),
                ReductionOp.Max => type.GetMinValue(),
                ReductionOp.Min => type.GetMaxValue(),
                ReductionOp.ArgMax => 0,
                ReductionOp.ArgMin => 0,
                ReductionOp.Mean => type.GetDefaultValue(),
                ReductionOp.CumSum => type.GetDefaultValue(),
                _ => throw new NotSupportedException($"Operation {op} has no identity element")
            };
        }

        /// <summary>
        /// Check if this reduction returns an index rather than a value.
        /// </summary>
        public static bool ReturnsIndex(this ReductionOp op)
        {
            return op == ReductionOp.ArgMax || op == ReductionOp.ArgMin;
        }

        /// <summary>
        /// Check if this reduction is order-dependent (cannot be parallelized trivially).
        /// </summary>
        public static bool IsOrderDependent(this ReductionOp op)
        {
            return op == ReductionOp.ArgMax || op == ReductionOp.ArgMin;
        }

        /// <summary>
        /// Check if this reduction has SIMD horizontal reduction support.
        /// </summary>
        public static bool HasSimdSupport(this ReductionOp op)
        {
            // Vector256 has Sum (horizontal add) but not Max/Min horizontal
            // For Max/Min we need to reduce the vector at the end
            return op == ReductionOp.Sum || op == ReductionOp.Max || op == ReductionOp.Min;
        }
    }

    /// <summary>
    /// Extension methods for NPTypeCode related to reductions.
    /// </summary>
    public static class ReductionTypeExtensions
    {
        /// <summary>
        /// Get the default value (additive identity) for a type.
        /// </summary>
        public static object GetDefaultValue(this NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => false,
                NPTypeCode.Byte => (byte)0,
                NPTypeCode.Int16 => (short)0,
                NPTypeCode.UInt16 => (ushort)0,
                NPTypeCode.Int32 => 0,
                NPTypeCode.UInt32 => 0u,
                NPTypeCode.Int64 => 0L,
                NPTypeCode.UInt64 => 0UL,
                NPTypeCode.Char => (char)0,
                NPTypeCode.Single => 0f,
                NPTypeCode.Double => 0d,
                NPTypeCode.Decimal => 0m,
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Get the multiplicative identity (1) for a type.
        /// </summary>
        public static object GetOneValue(this NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => true,
                NPTypeCode.Byte => (byte)1,
                NPTypeCode.Int16 => (short)1,
                NPTypeCode.UInt16 => (ushort)1,
                NPTypeCode.Int32 => 1,
                NPTypeCode.UInt32 => 1u,
                NPTypeCode.Int64 => 1L,
                NPTypeCode.UInt64 => 1UL,
                NPTypeCode.Char => (char)1,
                NPTypeCode.Single => 1f,
                NPTypeCode.Double => 1d,
                NPTypeCode.Decimal => 1m,
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Get the minimum value for a type (for Max reduction identity).
        /// </summary>
        public static object GetMinValue(this NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => false,
                NPTypeCode.Byte => byte.MinValue,
                NPTypeCode.Int16 => short.MinValue,
                NPTypeCode.UInt16 => ushort.MinValue,
                NPTypeCode.Int32 => int.MinValue,
                NPTypeCode.UInt32 => uint.MinValue,
                NPTypeCode.Int64 => long.MinValue,
                NPTypeCode.UInt64 => ulong.MinValue,
                NPTypeCode.Char => char.MinValue,
                NPTypeCode.Single => float.NegativeInfinity,
                NPTypeCode.Double => double.NegativeInfinity,
                NPTypeCode.Decimal => decimal.MinValue,
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Get the maximum value for a type (for Min reduction identity).
        /// </summary>
        public static object GetMaxValue(this NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => true,
                NPTypeCode.Byte => byte.MaxValue,
                NPTypeCode.Int16 => short.MaxValue,
                NPTypeCode.UInt16 => ushort.MaxValue,
                NPTypeCode.Int32 => int.MaxValue,
                NPTypeCode.UInt32 => uint.MaxValue,
                NPTypeCode.Int64 => long.MaxValue,
                NPTypeCode.UInt64 => ulong.MaxValue,
                NPTypeCode.Char => char.MaxValue,
                NPTypeCode.Single => float.PositiveInfinity,
                NPTypeCode.Double => double.PositiveInfinity,
                NPTypeCode.Decimal => decimal.MaxValue,
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }
    }
}
