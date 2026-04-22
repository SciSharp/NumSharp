using System;
using System.Numerics;

namespace NumSharp.Backends.Iteration
{
    public unsafe interface INpyAxisSameTypeKernel<T>
        where T : unmanaged
    {
        static abstract unsafe void Execute(T* src, T* dst, long srcStride, long dstStride, long length);
    }

    public readonly struct CumSumAxisKernel<T> : INpyAxisSameTypeKernel<T>
        where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
    {
        public static unsafe void Execute(T* src, T* dst, long srcStride, long dstStride, long length)
        {
            var sum = T.AdditiveIdentity;
            for (long i = 0; i < length; i++)
            {
                sum += src[i * srcStride];
                dst[i * dstStride] = sum;
            }
        }
    }

    public readonly struct CumProdAxisKernel<T> : INpyAxisSameTypeKernel<T>
        where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
    {
        public static unsafe void Execute(T* src, T* dst, long srcStride, long dstStride, long length)
        {
            var product = T.MultiplicativeIdentity;
            for (long i = 0; i < length; i++)
            {
                product *= src[i * srcStride];
                dst[i * dstStride] = product;
            }
        }
    }

    public interface INpyAxisDoubleReductionKernel
    {
        static abstract unsafe double Execute(double* src, long srcStride, long length, int ddof);
    }

    public readonly struct VarAxisDoubleKernel : INpyAxisDoubleReductionKernel
    {
        public static unsafe double Execute(double* src, long srcStride, long length, int ddof)
        {
            double sum = 0;
            for (long i = 0; i < length; i++)
                sum += src[i * srcStride];

            double mean = sum / length;
            double sq = 0;
            for (long i = 0; i < length; i++)
            {
                double value = src[i * srcStride] - mean;
                sq += value * value;
            }

            return sq / (length - ddof);
        }
    }

    public readonly struct StdAxisDoubleKernel : INpyAxisDoubleReductionKernel
    {
        public static unsafe double Execute(double* src, long srcStride, long length, int ddof)
            => Math.Sqrt(VarAxisDoubleKernel.Execute(src, srcStride, length, ddof));
    }

    public static unsafe class NpyAxisIter
    {
        public static void ExecuteSameType<T, TKernel>(UnmanagedStorage src, UnmanagedStorage dst, int axis)
            where T : unmanaged
            where TKernel : struct, INpyAxisSameTypeKernel<T>
        {
            var state = CreateState(src, dst, axis);
            if (state.AxisLength == 0 || state.OuterSize == 0)
                return;

            var srcBase = (T*)state.Data0;
            var dstBase = (T*)state.Data1;

            if (state.OuterNDim == 0)
            {
                TKernel.Execute(srcBase, dstBase, state.SourceAxisStride, state.DestinationAxisStride, state.AxisLength);
                return;
            }

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();
            var dstOuterStrides = state.GetDestinationOuterStridesPointer();

            for (long outerIndex = 0; outerIndex < state.OuterSize; outerIndex++)
            {
                long srcOffset = 0;
                long dstOffset = 0;
                long idx = outerIndex;

                for (int axisIndex = state.OuterNDim - 1; axisIndex >= 0; axisIndex--)
                {
                    long dim = outerShape[axisIndex];
                    long coord = idx % dim;
                    idx /= dim;

                    srcOffset += coord * srcOuterStrides[axisIndex];
                    dstOffset += coord * dstOuterStrides[axisIndex];
                }

                TKernel.Execute(
                    srcBase + srcOffset,
                    dstBase + dstOffset,
                    state.SourceAxisStride,
                    state.DestinationAxisStride,
                    state.AxisLength);
            }
        }

        public static void ReduceDouble<TKernel>(UnmanagedStorage src, UnmanagedStorage dst, int axis, int ddof)
            where TKernel : struct, INpyAxisDoubleReductionKernel
        {
            var state = CreateReductionState(src, dst, axis);
            if (state.AxisLength == 0 || state.OuterSize == 0)
                return;

            var srcBase = (double*)state.Data0;

            switch (dst.TypeCode)
            {
                case NPTypeCode.Single:
                {
                    var dstPtr = (float*)state.Data1;
                    ExecuteReductionLoopSingle<TKernel>(ref state, srcBase, dstPtr, ddof);
                    break;
                }
                case NPTypeCode.Double:
                {
                    var dstPtr = (double*)state.Data1;
                    ExecuteReductionLoopDouble<TKernel>(ref state, srcBase, dstPtr, ddof);
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    var dstPtr = (decimal*)state.Data1;
                    ExecuteReductionLoopDecimal<TKernel>(ref state, srcBase, dstPtr, ddof);
                    break;
                }
                default:
                    throw new NotSupportedException($"Axis reduction output type {dst.TypeCode} is not supported for double reductions.");
            }
        }

        public static void ReduceBool<T, TKernel>(UnmanagedStorage src, UnmanagedStorage dst, int axis)
            where T : unmanaged
            where TKernel : struct, INpyBooleanReductionKernel<T>
        {
            var state = CreateReductionState(src, dst, axis);
            if (state.OuterSize == 0)
                return;

            var dstBase = (bool*)state.Data1;

            if (state.AxisLength == 0)
            {
                FillBool(dstBase, state.OuterSize, TKernel.Identity);
                return;
            }

            var srcBase = (T*)state.Data0;
            if (state.OuterNDim == 0)
            {
                dstBase[0] = ExecuteBoolKernel<T, TKernel>(srcBase, state.SourceAxisStride, state.AxisLength);
                return;
            }

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();

            for (long outerIndex = 0; outerIndex < state.OuterSize; outerIndex++)
            {
                long srcOffset = 0;
                long idx = outerIndex;

                for (int axisIndex = state.OuterNDim - 1; axisIndex >= 0; axisIndex--)
                {
                    long dim = outerShape[axisIndex];
                    long coord = idx % dim;
                    idx /= dim;
                    srcOffset += coord * srcOuterStrides[axisIndex];
                }

                dstBase[outerIndex] = ExecuteBoolKernel<T, TKernel>(
                    srcBase + srcOffset,
                    state.SourceAxisStride,
                    state.AxisLength);
            }
        }

        private static NpyAxisState CreateState(UnmanagedStorage src, UnmanagedStorage dst, int axis)
        {
            if (src.Shape.NDim != dst.Shape.NDim)
                throw new NotSupportedException("NpyAxisIter currently requires source and destination to have matching ranks.");

            int ndim = checked((int)src.Shape.NDim);
            if (ndim > NpyAxisState.MaxDims)
                throw new NotSupportedException($"NpyAxisIter currently supports up to {NpyAxisState.MaxDims} dimensions.");

            if ((uint)axis >= (uint)ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            var state = new NpyAxisState
            {
                Axis = axis,
                AxisLength = src.Shape.dimensions[axis],
                SourceAxisStride = src.Shape.strides[axis],
                DestinationAxisStride = dst.Shape.strides[axis],
                Data0 = (IntPtr)((byte*)src.Address + (src.Shape.offset * src.InternalArray.ItemLength)),
                Data1 = (IntPtr)((byte*)dst.Address + (dst.Shape.offset * dst.InternalArray.ItemLength)),
            };

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();
            var dstOuterStrides = state.GetDestinationOuterStridesPointer();

            int outerAxis = 0;
            long outerSize = 1;
            for (int i = 0; i < ndim; i++)
            {
                if (i == axis)
                    continue;

                long dim = src.Shape.dimensions[i];
                if (dim == 0)
                {
                    state.OuterNDim = 0;
                    state.OuterSize = 0;
                    return state;
                }

                if (dim == 1)
                    continue;

                outerShape[outerAxis] = dim;
                srcOuterStrides[outerAxis] = src.Shape.strides[i];
                dstOuterStrides[outerAxis] = dst.Shape.strides[i];
                outerSize *= dim;
                outerAxis++;
            }

            state.OuterNDim = outerAxis;
            state.OuterSize = outerSize;

            if (state.OuterNDim == 0 && state.AxisLength > 0)
                state.OuterSize = 1;

            return state;
        }

        private static NpyAxisState CreateReductionState(UnmanagedStorage src, UnmanagedStorage dst, int axis)
        {
            int ndim = checked((int)src.Shape.NDim);
            if (ndim > NpyAxisState.MaxDims)
                throw new NotSupportedException($"NpyAxisIter currently supports up to {NpyAxisState.MaxDims} dimensions.");

            if ((uint)axis >= (uint)ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            var state = new NpyAxisState
            {
                Axis = axis,
                AxisLength = src.Shape.dimensions[axis],
                SourceAxisStride = src.Shape.strides[axis],
                Data0 = (IntPtr)((byte*)src.Address + (src.Shape.offset * src.InternalArray.ItemLength)),
                Data1 = (IntPtr)((byte*)dst.Address + (dst.Shape.offset * dst.InternalArray.ItemLength)),
            };

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();

            int outerAxis = 0;
            long outerSize = 1;
            for (int i = 0; i < ndim; i++)
            {
                if (i == axis)
                    continue;

                long dim = src.Shape.dimensions[i];
                if (dim == 0)
                {
                    state.OuterNDim = 0;
                    state.OuterSize = 0;
                    return state;
                }

                if (dim == 1)
                    continue;

                outerShape[outerAxis] = dim;
                srcOuterStrides[outerAxis] = src.Shape.strides[i];
                outerSize *= dim;
                outerAxis++;
            }

            state.OuterNDim = outerAxis;
            state.OuterSize = outerSize;

            if (state.OuterNDim == 0 && state.AxisLength > 0)
                state.OuterSize = 1;

            if (dst.Shape.IsContiguous && dst.Shape.size != state.OuterSize)
                throw new InvalidOperationException("Axis reduction output size does not match the iterator outer size.");

            return state;
        }

        private static void ExecuteReductionLoopSingle<TKernel>(
            ref NpyAxisState state,
            double* srcBase,
            float* dstBase,
            int ddof)
            where TKernel : struct, INpyAxisDoubleReductionKernel
        {
            if (state.OuterNDim == 0)
            {
                dstBase[0] = (float)TKernel.Execute(srcBase, state.SourceAxisStride, state.AxisLength, ddof);
                return;
            }

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();

            for (long outerIndex = 0; outerIndex < state.OuterSize; outerIndex++)
            {
                long srcOffset = 0;
                long idx = outerIndex;

                for (int axisIndex = state.OuterNDim - 1; axisIndex >= 0; axisIndex--)
                {
                    long dim = outerShape[axisIndex];
                    long coord = idx % dim;
                    idx /= dim;
                    srcOffset += coord * srcOuterStrides[axisIndex];
                }

                dstBase[outerIndex] = (float)TKernel.Execute(srcBase + srcOffset, state.SourceAxisStride, state.AxisLength, ddof);
            }
        }

        private static void ExecuteReductionLoopDouble<TKernel>(
            ref NpyAxisState state,
            double* srcBase,
            double* dstBase,
            int ddof)
            where TKernel : struct, INpyAxisDoubleReductionKernel
        {
            if (state.OuterNDim == 0)
            {
                dstBase[0] = TKernel.Execute(srcBase, state.SourceAxisStride, state.AxisLength, ddof);
                return;
            }

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();

            for (long outerIndex = 0; outerIndex < state.OuterSize; outerIndex++)
            {
                long srcOffset = 0;
                long idx = outerIndex;

                for (int axisIndex = state.OuterNDim - 1; axisIndex >= 0; axisIndex--)
                {
                    long dim = outerShape[axisIndex];
                    long coord = idx % dim;
                    idx /= dim;
                    srcOffset += coord * srcOuterStrides[axisIndex];
                }

                dstBase[outerIndex] = TKernel.Execute(srcBase + srcOffset, state.SourceAxisStride, state.AxisLength, ddof);
            }
        }

        private static void ExecuteReductionLoopDecimal<TKernel>(
            ref NpyAxisState state,
            double* srcBase,
            decimal* dstBase,
            int ddof)
            where TKernel : struct, INpyAxisDoubleReductionKernel
        {
            if (state.OuterNDim == 0)
            {
                dstBase[0] = (decimal)TKernel.Execute(srcBase, state.SourceAxisStride, state.AxisLength, ddof);
                return;
            }

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();

            for (long outerIndex = 0; outerIndex < state.OuterSize; outerIndex++)
            {
                long srcOffset = 0;
                long idx = outerIndex;

                for (int axisIndex = state.OuterNDim - 1; axisIndex >= 0; axisIndex--)
                {
                    long dim = outerShape[axisIndex];
                    long coord = idx % dim;
                    idx /= dim;
                    srcOffset += coord * srcOuterStrides[axisIndex];
                }

                dstBase[outerIndex] = (decimal)TKernel.Execute(srcBase + srcOffset, state.SourceAxisStride, state.AxisLength, ddof);
            }
        }

        private static bool ExecuteBoolKernel<T, TKernel>(T* src, long srcStride, long length)
            where T : unmanaged
            where TKernel : struct, INpyBooleanReductionKernel<T>
        {
            bool accumulator = TKernel.Identity;
            for (long i = 0; i < length; i++)
            {
                accumulator = TKernel.Accumulate(accumulator, src[i * srcStride]);
                if (TKernel.ShouldExit(accumulator))
                    break;
            }

            return accumulator;
        }

        private static void FillBool(bool* dst, long length, bool value)
        {
            for (long i = 0; i < length; i++)
                dst[i] = value;
        }

        // =========================================================================
        // Numeric Axis Reduction (sum, prod, min, max along axis)
        // =========================================================================

        /// <summary>
        /// Execute a numeric reduction along an axis using the provided kernel.
        /// Used as fallback for non-contiguous, sliced, or broadcast arrays.
        /// </summary>
        public static void ReduceNumeric<T, TKernel>(UnmanagedStorage src, UnmanagedStorage dst, int axis)
            where T : unmanaged
            where TKernel : struct, INpyAxisNumericReductionKernel<T>
        {
            var state = CreateReductionState(src, dst, axis);
            if (state.OuterSize == 0)
                return;

            var dstBase = (T*)state.Data1;

            if (state.AxisLength == 0)
            {
                // For empty axis, we need to set identity value based on operation
                // This is handled by caller before invoking this method
                return;
            }

            var srcBase = (T*)state.Data0;

            if (state.OuterNDim == 0)
            {
                dstBase[0] = TKernel.Execute(srcBase, state.SourceAxisStride, state.AxisLength);
                return;
            }

            var outerShape = state.GetOuterShapePointer();
            var srcOuterStrides = state.GetSourceOuterStridesPointer();

            for (long outerIndex = 0; outerIndex < state.OuterSize; outerIndex++)
            {
                long srcOffset = 0;
                long idx = outerIndex;

                for (int axisIndex = state.OuterNDim - 1; axisIndex >= 0; axisIndex--)
                {
                    long dim = outerShape[axisIndex];
                    long coord = idx % dim;
                    idx /= dim;
                    srcOffset += coord * srcOuterStrides[axisIndex];
                }

                dstBase[outerIndex] = TKernel.Execute(
                    srcBase + srcOffset,
                    state.SourceAxisStride,
                    state.AxisLength);
            }
        }
    }
}
