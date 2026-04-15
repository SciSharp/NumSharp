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
            // Detect scalar NDArrays (from implicit primitive conversion or explicit NDArray.Scalar)
            // Scalar NDArrays use NEP50 weak scalar type promotion rules
            bool xIsScalar = x.Shape.IsScalar;
            bool yIsScalar = y.Shape.IsScalar;
            return where_internal(condition, x, y, xIsScalar, yIsScalar);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for x.
        /// </summary>
        public static NDArray where(NDArray condition, object x, NDArray y)
        {
            var xArr = asanyarray(x);
            return where_internal(condition, xArr, y, xArr.Shape.IsScalar, y.Shape.IsScalar);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for y.
        /// </summary>
        public static NDArray where(NDArray condition, NDArray x, object y)
        {
            var yArr = asanyarray(y);
            return where_internal(condition, x, yArr, x.Shape.IsScalar, yArr.Shape.IsScalar);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for both x and y.
        /// </summary>
        public static NDArray where(NDArray condition, object x, object y)
        {
            var xArr = asanyarray(x);
            var yArr = asanyarray(y);
            return where_internal(condition, xArr, yArr, xArr.Shape.IsScalar, yArr.Shape.IsScalar);
        }

        /// <summary>
        /// Internal implementation of np.where with scalar tracking for NEP50 type promotion.
        /// </summary>
        /// <param name="condition">Condition array</param>
        /// <param name="x">X values (already converted to NDArray)</param>
        /// <param name="y">Y values (already converted to NDArray)</param>
        /// <param name="xIsScalar">True if x is a scalar NDArray</param>
        /// <param name="yIsScalar">True if y is a scalar NDArray</param>
        private static NDArray where_internal(NDArray condition, NDArray x, NDArray y, bool xIsScalar, bool yIsScalar)
        {
            // Broadcast all three arrays to common shape
            var broadcasted = broadcast_arrays(condition, x, y);
            var cond = broadcasted[0];
            var xArr = broadcasted[1];
            var yArr = broadcasted[2];

            // Determine output dtype from x and y using NEP50-aware type promotion
            var outType = _FindCommonTypeForWhere(x.GetTypeCode, y.GetTypeCode, xIsScalar, yIsScalar);

            // Convert x and y to output type if needed (required for kernel and iterator paths)
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

        /// <summary>
        /// Determines the output dtype for np.where following NumPy 2.x NEP50 rules.
        ///
        /// Rules:
        /// 1. Both arrays (non-scalar): use array-array promotion table
        /// 2. Both were scalars: use Python-like defaults (int32→int64)
        /// 3. One array, one scalar: use NEP50 weak scalar rules (array dtype wins for same-kind)
        /// </summary>
        private static NPTypeCode _FindCommonTypeForWhere(NPTypeCode xType, NPTypeCode yType, bool xIsScalar, bool yIsScalar)
        {
            // Case 1: Both are scalars - use Python-like default type widening
            if (xIsScalar && yIsScalar)
            {
                return _GetPythonLikeScalarType(xType, yType);
            }

            // Case 2: One is scalar, one is array - use NEP50 weak scalar rules
            if (xIsScalar)
            {
                // y is array, x is scalar - array wins for same-kind
                return _FindCommonArrayScalarType(yType, xType);
            }
            if (yIsScalar)
            {
                // x is array, y is scalar - array wins for same-kind
                return _FindCommonArrayScalarType(xType, yType);
            }

            // Case 3: Both are arrays - use array-array promotion
            return _FindCommonArrayType(xType, yType);
        }

        /// <summary>
        /// Determines the result type when both operands are scalar NDArrays.
        ///
        /// C# limitation: We cannot distinguish between:
        /// - `np.where(cond, 1, 0)` where 1,0 are C# int literals (implicit conversion)
        /// - `np.where(cond, np.array(1), np.array(0))` where arrays are explicitly created
        ///
        /// Both cases create int32 scalar NDArrays. We preserve the type when both
        /// scalars are the same type, and use NEP50 weak scalar rules otherwise.
        /// This differs from NumPy where Python int literals widen to int64.
        /// </summary>
        private static NPTypeCode _GetPythonLikeScalarType(NPTypeCode xType, NPTypeCode yType)
        {
            // Same type: preserve it (no widening)
            // This handles np.where(cond, 1, 0) → int32, np.where(cond, 1L, 0L) → int64
            if (xType == yType)
                return xType;

            // Different types - apply promotion rules
            var xKind = GetTypeKind(xType);
            var yKind = GetTypeKind(yType);

            // Cross-kind promotion: use standard array-array rules
            if (xKind != yKind)
            {
                return _FindCommonArrayType(xType, yType);
            }

            // Same kind, different types - use array-array promotion
            return _FindCommonArrayType(xType, yType);
        }

        /// <summary>
        /// Returns the kind character for a type (matching NumPy's dtype.kind).
        /// </summary>
        private static char GetTypeKind(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => 'b',
                NPTypeCode.Byte => 'u',
                NPTypeCode.UInt16 => 'u',
                NPTypeCode.UInt32 => 'u',
                NPTypeCode.UInt64 => 'u',
                NPTypeCode.Int16 => 'i',
                NPTypeCode.Int32 => 'i',
                NPTypeCode.Int64 => 'i',
                NPTypeCode.Char => 'u', // char is essentially uint16
                NPTypeCode.Single => 'f',
                NPTypeCode.Double => 'f',
                NPTypeCode.Decimal => 'f', // treat decimal as float-like
                NPTypeCode.Complex => 'c',
                _ => '?'
            };
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
