using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Equivalent to <see cref="nonzero(NDArray)"/>: returns the indices where
        ///     <paramref name="condition"/> is non-zero.
        /// </summary>
        /// <param name="condition">Input array. Non-zero entries yield their indices.</param>
        /// <returns>Tuple of arrays with indices where condition is non-zero, one per dimension.</returns>
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
            return where_internal(condition, x, y);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for x.
        /// </summary>
        public static NDArray where(NDArray condition, object x, NDArray y)
        {
            return where_internal(condition, asanyarray(x), y);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for y.
        /// </summary>
        public static NDArray where(NDArray condition, NDArray x, object y)
        {
            return where_internal(condition, x, asanyarray(y));
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for both x and y.
        /// </summary>
        public static NDArray where(NDArray condition, object x, object y)
        {
            return where_internal(condition, asanyarray(x), asanyarray(y));
        }

        /// <summary>
        /// Internal implementation of np.where.
        /// </summary>
        private static NDArray where_internal(NDArray condition, NDArray x, NDArray y)
        {
            // Skip broadcast_arrays (which allocates 3 NDArrays + helper arrays) when all three
            // already share a shape — the frequent case of np.where(mask, arr, other_arr).
            NDArray cond, xArr, yArr;
            if (condition.Shape == x.Shape && x.Shape == y.Shape)
            {
                cond = condition;
                xArr = x;
                yArr = y;
            }
            else
            {
                var broadcasted = broadcast_arrays(condition, x, y);
                cond = broadcasted[0];
                xArr = broadcasted[1];
                yArr = broadcasted[2];
            }

            // Coerce the condition to boolean using NumPy's truthiness rules
            // (0/0.0 → False, everything else including NaN/±Inf → True). The
            // iterator-driven expression kernel requires a bool condition dtype.
            if (cond.GetTypeCode != NPTypeCode.Boolean)
                cond = cond.astype(NPTypeCode.Boolean, copy: false);

            // When x and y already agree, skip the NEP50 promotion lookup. Otherwise defer to
            // _FindCommonType which handles the scalar+array NEP50 rules.
            var outType = x.GetTypeCode == y.GetTypeCode
                ? x.GetTypeCode
                : _FindCommonType(x, y);

            if (xArr.GetTypeCode != outType)
                xArr = xArr.astype(outType, copy: false);
            if (yArr.GetTypeCode != outType)
                yArr = yArr.astype(outType, copy: false);

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

        private static void WhereImpl<T>(NDArray cond, NDArray x, NDArray y, NDArray result) where T : unmanaged
        {
            // Drive cond + x + y + result in lockstep via a 4-operand NpyIter
            // compiling Where(cond, x, y) → out as a single IL expression kernel.
            // C-order traversal matches NumPy element semantics; WRITEONLY on
            // the output lets the iterator allocate per-inner-loop buffer space
            // when casting is needed.
            var dtype = result.GetTypeCode;
            using var iter = NpyIterRef.MultiNew(
                4, new[] { cond, x, y, result },
                NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY,
                });

            var expr = NpyExpr.Where(NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2));
            iter.ExecuteExpression(
                expr,
                new[] { NPTypeCode.Boolean, dtype, dtype },
                dtype,
                cacheKey: $"np.where.{dtype}");
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
                default:
                    throw new NotSupportedException($"Type {outType} not supported for np.where");
            }
        }
    }
}
