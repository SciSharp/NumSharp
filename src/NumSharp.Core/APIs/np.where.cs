using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        /// </summary>
        /// <param name="condition">Where True, yield `x`, otherwise yield `y`.</param>
        /// <returns>Tuple of arrays with indices where condition is non-zero (equivalent to np.nonzero).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.where.html</remarks>
        public static NDArray<long>[] where(NDArray condition)
        {
            return nonzero(condition);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        /// </summary>
        /// <param name="condition">Where True, yield `x`, otherwise yield `y`.</param>
        /// <param name="x">Values from which to choose where condition is True.</param>
        /// <param name="y">Values from which to choose where condition is False.</param>
        /// <returns>An array with elements from `x` where `condition` is True, and elements from `y` elsewhere.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.where.html</remarks>
        public static NDArray where(NDArray condition, NDArray x, NDArray y)
        {
            // Broadcast all three arrays to common shape
            var broadcasted = broadcast_arrays(condition, x, y);
            var cond = broadcasted[0];
            var xArr = broadcasted[1];
            var yArr = broadcasted[2];

            // Determine output dtype from x and y (type promotion)
            var outType = _FindCommonType(xArr, yArr);
            // Use cond.shape (dimensions only) not cond.Shape (which may have broadcast strides)
            var result = empty(cond.shape, outType);

            // Handle empty arrays - nothing to iterate
            if (result.size == 0)
                return result;

            // IL Kernel fast path: all arrays contiguous, bool condition, SIMD enabled
            // Broadcasted arrays (stride=0) are NOT contiguous, so they use iterator path.
            bool canUseKernel = ILKernelGenerator.Enabled &&
                                cond.typecode == NPTypeCode.Boolean &&
                                cond.Shape.IsContiguous &&
                                xArr.Shape.IsContiguous &&
                                yArr.Shape.IsContiguous;

            if (canUseKernel)
            {
                WhereKernelDispatch(cond, xArr, yArr, result, outType);
                return result;
            }

            // Iterator fallback for non-contiguous/broadcasted arrays
            switch (outType)
            {
                case NPTypeCode.Boolean:
                    WhereImpl<bool>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Byte:
                    WhereImpl<byte>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Int16:
                    WhereImpl<short>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.UInt16:
                    WhereImpl<ushort>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Int32:
                    WhereImpl<int>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.UInt32:
                    WhereImpl<uint>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Int64:
                    WhereImpl<long>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.UInt64:
                    WhereImpl<ulong>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Char:
                    WhereImpl<char>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Single:
                    WhereImpl<float>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Double:
                    WhereImpl<double>(cond, xArr, yArr, result);
                    break;
                case NPTypeCode.Decimal:
                    WhereImpl<decimal>(cond, xArr, yArr, result);
                    break;
                default:
                    throw new NotSupportedException($"Type {outType} not supported for np.where");
            }

            return result;
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for x.
        /// </summary>
        public static NDArray where(NDArray condition, object x, NDArray y)
        {
            return where(condition, asanyarray(x), y);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for y.
        /// </summary>
        public static NDArray where(NDArray condition, NDArray x, object y)
        {
            return where(condition, x, asanyarray(y));
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for both x and y.
        /// </summary>
        public static NDArray where(NDArray condition, object x, object y)
        {
            return where(condition, asanyarray(x), asanyarray(y));
        }

        private static void WhereImpl<T>(NDArray cond, NDArray x, NDArray y, NDArray result) where T : unmanaged
        {
            // Use iterators for proper handling of broadcasted/strided arrays
            using var condIter = cond.AsIterator<bool>();
            using var xIter = x.AsIterator<T>();
            using var yIter = y.AsIterator<T>();
            using var resultIter = result.AsIterator<T>();

            while (condIter.HasNext())
            {
                var c = condIter.MoveNext();
                var xVal = xIter.MoveNext();
                var yVal = yIter.MoveNext();
                resultIter.MoveNextReference() = c ? xVal : yVal;
            }
        }

        /// <summary>
        /// IL Kernel dispatch for contiguous arrays.
        /// Uses IL-generated kernels with SIMD optimization.
        /// </summary>
        private static unsafe void WhereKernelDispatch(NDArray cond, NDArray x, NDArray y, NDArray result, NPTypeCode outType)
        {
            var condPtr = (bool*)cond.Address;
            var count = result.size;

            switch (outType)
            {
                case NPTypeCode.Boolean:
                    ILKernelGenerator.WhereExecute(condPtr, (bool*)x.Address, (bool*)y.Address, (bool*)result.Address, count);
                    break;
                case NPTypeCode.Byte:
                    ILKernelGenerator.WhereExecute(condPtr, (byte*)x.Address, (byte*)y.Address, (byte*)result.Address, count);
                    break;
                case NPTypeCode.Int16:
                    ILKernelGenerator.WhereExecute(condPtr, (short*)x.Address, (short*)y.Address, (short*)result.Address, count);
                    break;
                case NPTypeCode.UInt16:
                    ILKernelGenerator.WhereExecute(condPtr, (ushort*)x.Address, (ushort*)y.Address, (ushort*)result.Address, count);
                    break;
                case NPTypeCode.Int32:
                    ILKernelGenerator.WhereExecute(condPtr, (int*)x.Address, (int*)y.Address, (int*)result.Address, count);
                    break;
                case NPTypeCode.UInt32:
                    ILKernelGenerator.WhereExecute(condPtr, (uint*)x.Address, (uint*)y.Address, (uint*)result.Address, count);
                    break;
                case NPTypeCode.Int64:
                    ILKernelGenerator.WhereExecute(condPtr, (long*)x.Address, (long*)y.Address, (long*)result.Address, count);
                    break;
                case NPTypeCode.UInt64:
                    ILKernelGenerator.WhereExecute(condPtr, (ulong*)x.Address, (ulong*)y.Address, (ulong*)result.Address, count);
                    break;
                case NPTypeCode.Char:
                    ILKernelGenerator.WhereExecute(condPtr, (char*)x.Address, (char*)y.Address, (char*)result.Address, count);
                    break;
                case NPTypeCode.Single:
                    ILKernelGenerator.WhereExecute(condPtr, (float*)x.Address, (float*)y.Address, (float*)result.Address, count);
                    break;
                case NPTypeCode.Double:
                    ILKernelGenerator.WhereExecute(condPtr, (double*)x.Address, (double*)y.Address, (double*)result.Address, count);
                    break;
                case NPTypeCode.Decimal:
                    ILKernelGenerator.WhereExecute(condPtr, (decimal*)x.Address, (decimal*)y.Address, (decimal*)result.Address, count);
                    break;
            }
        }
    }
}
