using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Reduction operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute an element-wise reduction operation (axis=null) using IL-generated kernels.
        /// Reduces all elements to a single scalar value.
        /// </summary>
        /// <typeparam name="TResult">Result type</typeparam>
        /// <param name="arr">Input array</param>
        /// <param name="op">Reduction operation</param>
        /// <param name="accumulatorType">Optional accumulator type (defaults to input type)</param>
        /// <returns>Scalar result</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe TResult ExecuteElementReduction<TResult>(in NDArray arr, ReductionOp op, NPTypeCode? accumulatorType = null)
            where TResult : unmanaged
        {
            if (arr.size == 0)
            {
                // Return identity for empty arrays
                return (TResult)op.GetIdentity(typeof(TResult).GetTypeCode());
            }

            var inputType = arr.GetTypeCode;
            var accumType = accumulatorType ?? inputType.GetAccumulatingType();

            // Handle scalar case - just return the value (possibly converted)
            if (arr.Shape.IsScalar)
            {
                return ExecuteScalarReduction<TResult>(arr, op, accumType);
            }

            // Determine if array is contiguous
            bool isContiguous = arr.Shape.IsContiguous;

            // Get kernel key
            var key = new ElementReductionKernelKey(inputType, accumType, op, isContiguous);

            // Get or generate kernel
            var kernel = ILKernelGenerator.TryGetTypedElementReductionKernel<TResult>(key);

            if (kernel != null)
            {
                return ExecuteTypedReductionKernel<TResult>(kernel, arr);
            }
            else
            {
                // Fallback - should not happen for implemented operations
                throw new NotSupportedException(
                    $"IL kernel not available for {op}({inputType}) -> {accumType}. " +
                    "Please report this as a bug.");
            }
        }

        /// <summary>
        /// Execute scalar reduction - just return the value, possibly converted.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TResult ExecuteScalarReduction<TResult>(in NDArray arr, ReductionOp op, NPTypeCode accumType)
            where TResult : unmanaged
        {
            // For ArgMax/ArgMin of scalar, index is 0
            if (op == ReductionOp.ArgMax || op == ReductionOp.ArgMin)
            {
                return (TResult)(object)0;
            }

            // For other ops, return the scalar value converted to result type
            var value = arr.GetAtIndex(0);
            return (TResult)Converts.ChangeType(value, typeof(TResult).GetTypeCode());
        }

        /// <summary>
        /// Execute the IL-generated typed reduction kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe TResult ExecuteTypedReductionKernel<TResult>(
            TypedElementReductionKernel<TResult> kernel,
            in NDArray input)
            where TResult : unmanaged
        {
            int inputElemSize = input.dtypesize;
            var inputShape = input.Shape;

            // Calculate base address accounting for shape offset (for sliced views)
            byte* inputAddr = (byte*)input.Address + inputShape.offset * inputElemSize;

            fixed (int* strides = inputShape.strides)
            fixed (int* shape = inputShape.dimensions)
            {
                return kernel(
                    (void*)inputAddr,
                    strides,
                    shape,
                    input.ndim,
                    input.size
                );
            }
        }

        #region Type-Specific Element Reduction Wrappers

        /// <summary>
        /// Execute element-wise sum reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected object sum_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Sum, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Sum, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Sum, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Sum, retType),
                _ => throw new NotSupportedException($"Sum not supported for type {retType}")
            };
        }

        /// <summary>
        /// Execute element-wise product reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected object prod_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Prod, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Prod, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Prod, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Prod, retType),
                _ => throw new NotSupportedException($"Prod not supported for type {retType}")
            };
        }

        /// <summary>
        /// Execute element-wise max reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected object max_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode;

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Max, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Max, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Max, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Max, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Max, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Max, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Max, retType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Max, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Max, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Max, retType),
                _ => throw new NotSupportedException($"Max not supported for type {retType}")
            };
        }

        /// <summary>
        /// Execute element-wise min reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected object min_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode;

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Min, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Min, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Min, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Min, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Min, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Min, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Min, retType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Min, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Min, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Min, retType),
                _ => throw new NotSupportedException($"Min not supported for type {retType}")
            };
        }

        /// <summary>
        /// Execute element-wise argmax reduction using IL kernels.
        /// Returns the index of the maximum value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected int argmax_elementwise_il(NDArray arr)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return 0;

            var inputType = arr.GetTypeCode;

            // ArgMax always returns int, but needs accumulator type for comparison
            return inputType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.Byte),
                NPTypeCode.Int16 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.Int16),
                NPTypeCode.UInt16 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.UInt16),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.Int32),
                NPTypeCode.UInt32 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.UInt32),
                NPTypeCode.Int64 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.Int64),
                NPTypeCode.UInt64 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.UInt64),
                NPTypeCode.Single => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.Single),
                NPTypeCode.Double => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.Double),
                NPTypeCode.Decimal => ExecuteElementReduction<int>(arr, ReductionOp.ArgMax, NPTypeCode.Decimal),
                _ => throw new NotSupportedException($"ArgMax not supported for type {inputType}")
            };
        }

        /// <summary>
        /// Execute element-wise argmin reduction using IL kernels.
        /// Returns the index of the minimum value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected int argmin_elementwise_il(NDArray arr)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return 0;

            var inputType = arr.GetTypeCode;

            // ArgMin always returns int, but needs accumulator type for comparison
            return inputType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.Byte),
                NPTypeCode.Int16 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.Int16),
                NPTypeCode.UInt16 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.UInt16),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.Int32),
                NPTypeCode.UInt32 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.UInt32),
                NPTypeCode.Int64 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.Int64),
                NPTypeCode.UInt64 => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.UInt64),
                NPTypeCode.Single => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.Single),
                NPTypeCode.Double => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.Double),
                NPTypeCode.Decimal => ExecuteElementReduction<int>(arr, ReductionOp.ArgMin, NPTypeCode.Decimal),
                _ => throw new NotSupportedException($"ArgMin not supported for type {inputType}")
            };
        }

        /// <summary>
        /// Execute element-wise mean using IL kernels for sum.
        /// Mean = Sum / count
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected object mean_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
            {
                var val = arr.GetAtIndex(0);
                return typeCode.HasValue ? Converts.ChangeType(val, typeCode.Value) : Convert.ToDouble(val);
            }

            // Mean always computes in double for precision
            var retType = typeCode ?? NPTypeCode.Double;
            int count = arr.size;

            // Sum in accumulating type, then divide
            var sumType = arr.GetTypeCode.GetAccumulatingType();

            double sum = sumType switch
            {
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Decimal => (double)ExecuteElementReduction<decimal>(arr, ReductionOp.Sum, sumType),
                _ => throw new NotSupportedException($"Mean not supported for accumulator type {sumType}")
            };

            double mean = sum / count;
            return Converts.ChangeType(mean, retType);
        }

        #endregion
    }
}
