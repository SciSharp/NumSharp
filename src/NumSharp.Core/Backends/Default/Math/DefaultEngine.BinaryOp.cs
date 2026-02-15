using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Binary operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute a binary operation using IL-generated kernels.
        /// Handles type promotion, broadcasting, and kernel dispatch.
        /// </summary>
        /// <param name="lhs">Left operand</param>
        /// <param name="rhs">Right operand</param>
        /// <param name="op">Operation to perform</param>
        /// <returns>Result array with promoted type</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray ExecuteBinaryOp(in NDArray lhs, in NDArray rhs, BinaryOp op)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;

            // Determine result type using NumPy type promotion rules
            var resultType = np._FindCommonType(lhs, rhs);

            // NumPy: true division (/) always returns float64 for integer types
            // This matches Python 3 / NumPy 2.x semantics where / is "true division"
            // Group 3 = float (Single, Double), Group 4 = Decimal
            if (op == BinaryOp.Divide && resultType.GetGroup() < 3)
            {
                resultType = NPTypeCode.Double;
            }

            // Handle scalar × scalar case
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
            {
                return ExecuteScalarScalar(lhs, rhs, op, resultType);
            }

            // Broadcast shapes
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var resultShape = leftShape.Clean();

            // Allocate result
            var result = new NDArray(resultType, resultShape, false);

            // Classify execution path using strides
            ExecutionPath path;
            fixed (int* lhsStrides = leftShape.strides)
            fixed (int* rhsStrides = rightShape.strides)
            fixed (int* shape = resultShape.dimensions)
            {
                path = ClassifyPath(lhsStrides, rhsStrides, shape, resultShape.NDim, resultType);
            }

            // Get kernel key
            var key = new MixedTypeKernelKey(lhsType, rhsType, resultType, op, path);

            // Get or generate kernel
            var kernel = ILKernelGenerator.TryGetMixedTypeKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel
                ExecuteKernel(kernel, lhs, rhs, result, leftShape, rightShape);
            }
            else
            {
                // Fallback to legacy implementation
                FallbackBinaryOp(lhs, rhs, result, op, leftShape, rightShape);
            }

            return result;
        }

        /// <summary>
        /// Execute scalar × scalar operation using IL-generated delegate.
        /// </summary>
        private NDArray ExecuteScalarScalar(in NDArray lhs, in NDArray rhs, BinaryOp op, NPTypeCode resultType)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;
            var key = new BinaryScalarKernelKey(lhsType, rhsType, resultType, op);
            var func = ILKernelGenerator.GetBinaryScalarDelegate(key);

            // Dispatch based on lhs type first
            return lhsType switch
            {
                NPTypeCode.Boolean => InvokeBinaryScalarLhs(func, lhs.GetBoolean(), rhs, rhsType, resultType),
                NPTypeCode.Byte => InvokeBinaryScalarLhs(func, lhs.GetByte(), rhs, rhsType, resultType),
                NPTypeCode.Int16 => InvokeBinaryScalarLhs(func, lhs.GetInt16(), rhs, rhsType, resultType),
                NPTypeCode.UInt16 => InvokeBinaryScalarLhs(func, lhs.GetUInt16(), rhs, rhsType, resultType),
                NPTypeCode.Int32 => InvokeBinaryScalarLhs(func, lhs.GetInt32(), rhs, rhsType, resultType),
                NPTypeCode.UInt32 => InvokeBinaryScalarLhs(func, lhs.GetUInt32(), rhs, rhsType, resultType),
                NPTypeCode.Int64 => InvokeBinaryScalarLhs(func, lhs.GetInt64(), rhs, rhsType, resultType),
                NPTypeCode.UInt64 => InvokeBinaryScalarLhs(func, lhs.GetUInt64(), rhs, rhsType, resultType),
                NPTypeCode.Char => InvokeBinaryScalarLhs(func, lhs.GetChar(), rhs, rhsType, resultType),
                NPTypeCode.Single => InvokeBinaryScalarLhs(func, lhs.GetSingle(), rhs, rhsType, resultType),
                NPTypeCode.Double => InvokeBinaryScalarLhs(func, lhs.GetDouble(), rhs, rhsType, resultType),
                NPTypeCode.Decimal => InvokeBinaryScalarLhs(func, lhs.GetDecimal(), rhs, rhsType, resultType),
                _ => throw new NotSupportedException($"LHS type {lhsType} not supported")
            };
        }

        /// <summary>
        /// Continue binary scalar dispatch with typed LHS value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray InvokeBinaryScalarLhs<TLhs>(
            Delegate func, TLhs lhsVal, in NDArray rhs, NPTypeCode rhsType, NPTypeCode resultType)
        {
            // Dispatch based on rhs type
            return rhsType switch
            {
                NPTypeCode.Boolean => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetBoolean(), resultType),
                NPTypeCode.Byte => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetByte(), resultType),
                NPTypeCode.Int16 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt16(), resultType),
                NPTypeCode.UInt16 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt16(), resultType),
                NPTypeCode.Int32 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt32(), resultType),
                NPTypeCode.UInt32 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt32(), resultType),
                NPTypeCode.Int64 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt64(), resultType),
                NPTypeCode.UInt64 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt64(), resultType),
                NPTypeCode.Char => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetChar(), resultType),
                NPTypeCode.Single => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetSingle(), resultType),
                NPTypeCode.Double => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetDouble(), resultType),
                NPTypeCode.Decimal => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetDecimal(), resultType),
                _ => throw new NotSupportedException($"RHS type {rhsType} not supported")
            };
        }

        /// <summary>
        /// Complete binary scalar dispatch with typed LHS and RHS values.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray InvokeBinaryScalarRhs<TLhs, TRhs>(
            Delegate func, TLhs lhsVal, TRhs rhsVal, NPTypeCode resultType)
        {
            // Dispatch based on result type
            return resultType switch
            {
                NPTypeCode.Boolean => NDArray.Scalar(((Func<TLhs, TRhs, bool>)func)(lhsVal, rhsVal)),
                NPTypeCode.Byte => NDArray.Scalar(((Func<TLhs, TRhs, byte>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int16 => NDArray.Scalar(((Func<TLhs, TRhs, short>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt16 => NDArray.Scalar(((Func<TLhs, TRhs, ushort>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int32 => NDArray.Scalar(((Func<TLhs, TRhs, int>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt32 => NDArray.Scalar(((Func<TLhs, TRhs, uint>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int64 => NDArray.Scalar(((Func<TLhs, TRhs, long>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt64 => NDArray.Scalar(((Func<TLhs, TRhs, ulong>)func)(lhsVal, rhsVal)),
                NPTypeCode.Char => NDArray.Scalar(((Func<TLhs, TRhs, char>)func)(lhsVal, rhsVal)),
                NPTypeCode.Single => NDArray.Scalar(((Func<TLhs, TRhs, float>)func)(lhsVal, rhsVal)),
                NPTypeCode.Double => NDArray.Scalar(((Func<TLhs, TRhs, double>)func)(lhsVal, rhsVal)),
                NPTypeCode.Decimal => NDArray.Scalar(((Func<TLhs, TRhs, decimal>)func)(lhsVal, rhsVal)),
                _ => throw new NotSupportedException($"Result type {resultType} not supported")
            };
        }

        /// <summary>
        /// Classify execution path based on strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ExecutionPath ClassifyPath(
            int* lhsStrides, int* rhsStrides, int* shape, int ndim, NPTypeCode resultType)
        {
            if (ndim == 0)
                return ExecutionPath.SimdFull;

            bool lhsContiguous = StrideDetector.IsContiguous(lhsStrides, shape, ndim);
            bool rhsContiguous = StrideDetector.IsContiguous(rhsStrides, shape, ndim);

            if (lhsContiguous && rhsContiguous)
                return ExecutionPath.SimdFull;

            // SimdScalarRight/Left require the non-scalar operand to be contiguous
            // because their loops use simple i * elemSize indexing
            bool rhsScalar = StrideDetector.IsScalar(rhsStrides, ndim);
            if (rhsScalar && lhsContiguous)
                return ExecutionPath.SimdScalarRight;

            bool lhsScalar = StrideDetector.IsScalar(lhsStrides, ndim);
            if (lhsScalar && rhsContiguous)
                return ExecutionPath.SimdScalarLeft;

            // Check for inner-contiguous (chunk-able)
            int lhsInner = lhsStrides[ndim - 1];
            int rhsInner = rhsStrides[ndim - 1];
            if ((lhsInner == 1 || lhsInner == 0) && (rhsInner == 1 || rhsInner == 0))
                return ExecutionPath.SimdChunk;

            return ExecutionPath.General;
        }

        /// <summary>
        /// Execute the IL-generated kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ExecuteKernel(
            MixedTypeKernel kernel,
            in NDArray lhs, in NDArray rhs, NDArray result,
            Shape lhsShape, Shape rhsShape)
        {
            // Get element sizes for offset calculation
            int lhsElemSize = lhs.dtypesize;
            int rhsElemSize = rhs.dtypesize;

            // Calculate base addresses accounting for shape offsets (for sliced views)
            // The Shape.offset represents the element offset into the underlying storage
            byte* lhsAddr = (byte*)lhs.Address + lhsShape.offset * lhsElemSize;
            byte* rhsAddr = (byte*)rhs.Address + rhsShape.offset * rhsElemSize;

            fixed (int* lhsStrides = lhsShape.strides)
            fixed (int* rhsStrides = rhsShape.strides)
            fixed (int* shape = result.shape)
            {
                kernel(
                    (void*)lhsAddr,
                    (void*)rhsAddr,
                    (void*)result.Address,
                    lhsStrides,
                    rhsStrides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }

        /// <summary>
        /// Fallback to legacy implementation when IL kernel is not available.
        /// </summary>
        private void FallbackBinaryOp(
            in NDArray lhs, in NDArray rhs, NDArray result,
            BinaryOp op, Shape lhsShape, Shape rhsShape)
        {
            // For now, throw - all kernels should be generatable
            // In future, this could call the legacy generated code
            throw new NotSupportedException(
                $"IL kernel not available for {lhs.GetTypeCode} {op} {rhs.GetTypeCode} -> {result.GetTypeCode}. " +
                "Please report this as a bug.");
        }
    }
}
