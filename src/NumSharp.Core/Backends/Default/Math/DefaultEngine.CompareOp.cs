using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Comparison operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute a comparison operation using IL-generated kernels.
        /// Handles type promotion, broadcasting, and kernel dispatch.
        /// Result is always NDArray&lt;bool&gt;.
        /// </summary>
        /// <param name="lhs">Left operand</param>
        /// <param name="rhs">Right operand</param>
        /// <param name="op">Comparison operation to perform</param>
        /// <returns>Result array with bool type</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray<bool> ExecuteComparisonOp(in NDArray lhs, in NDArray rhs, ComparisonOp op)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;

            // Handle scalar × scalar case
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
            {
                return ExecuteComparisonScalarScalar(lhs, rhs, op);
            }

            // Broadcast shapes
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var resultShape = leftShape.Clean();

            // Allocate result (always bool)
            var result = new NDArray<bool>(resultShape, true);

            // Classify execution path using strides
            ExecutionPath path;
            fixed (int* lhsStrides = leftShape.strides)
            fixed (int* rhsStrides = rightShape.strides)
            fixed (int* shape = resultShape.dimensions)
            {
                path = ClassifyPath(lhsStrides, rhsStrides, shape, resultShape.NDim, NPTypeCode.Boolean);
            }

            // Get kernel key
            var key = new ComparisonKernelKey(lhsType, rhsType, op, path);

            // Get or generate kernel
            var kernel = ILKernelGenerator.TryGetComparisonKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel
                ExecuteComparisonKernel(kernel, lhs, rhs, result, leftShape, rightShape);
            }
            else
            {
                // Fallback - should not happen
                throw new NotSupportedException(
                    $"IL kernel not available for comparison {lhsType} {op} {rhsType}. " +
                    "Please report this as a bug.");
            }

            return result;
        }

        /// <summary>
        /// Execute scalar × scalar comparison using IL-generated delegate.
        /// </summary>
        private NDArray<bool> ExecuteComparisonScalarScalar(in NDArray lhs, in NDArray rhs, ComparisonOp op)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;
            var key = new ILKernelGenerator.ComparisonScalarKernelKey(lhsType, rhsType, op);
            var func = ILKernelGenerator.GetComparisonScalarDelegate(key);

            // Dispatch based on lhs type first
            return lhsType switch
            {
                NPTypeCode.Boolean => InvokeComparisonScalarLhs(func, lhs.GetBoolean(), rhs, rhsType),
                NPTypeCode.Byte => InvokeComparisonScalarLhs(func, lhs.GetByte(), rhs, rhsType),
                NPTypeCode.Int16 => InvokeComparisonScalarLhs(func, lhs.GetInt16(), rhs, rhsType),
                NPTypeCode.UInt16 => InvokeComparisonScalarLhs(func, lhs.GetUInt16(), rhs, rhsType),
                NPTypeCode.Int32 => InvokeComparisonScalarLhs(func, lhs.GetInt32(), rhs, rhsType),
                NPTypeCode.UInt32 => InvokeComparisonScalarLhs(func, lhs.GetUInt32(), rhs, rhsType),
                NPTypeCode.Int64 => InvokeComparisonScalarLhs(func, lhs.GetInt64(), rhs, rhsType),
                NPTypeCode.UInt64 => InvokeComparisonScalarLhs(func, lhs.GetUInt64(), rhs, rhsType),
                NPTypeCode.Char => InvokeComparisonScalarLhs(func, lhs.GetChar(), rhs, rhsType),
                NPTypeCode.Single => InvokeComparisonScalarLhs(func, lhs.GetSingle(), rhs, rhsType),
                NPTypeCode.Double => InvokeComparisonScalarLhs(func, lhs.GetDouble(), rhs, rhsType),
                NPTypeCode.Decimal => InvokeComparisonScalarLhs(func, lhs.GetDecimal(), rhs, rhsType),
                _ => throw new NotSupportedException($"LHS type {lhsType} not supported")
            };
        }

        /// <summary>
        /// Continue comparison scalar dispatch with typed LHS value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray<bool> InvokeComparisonScalarLhs<TLhs>(
            Delegate func, TLhs lhsVal, in NDArray rhs, NPTypeCode rhsType)
        {
            // Dispatch based on rhs type
            return rhsType switch
            {
                NPTypeCode.Boolean => NDArray.Scalar(((Func<TLhs, bool, bool>)func)(lhsVal, rhs.GetBoolean())).MakeGeneric<bool>(),
                NPTypeCode.Byte => NDArray.Scalar(((Func<TLhs, byte, bool>)func)(lhsVal, rhs.GetByte())).MakeGeneric<bool>(),
                NPTypeCode.Int16 => NDArray.Scalar(((Func<TLhs, short, bool>)func)(lhsVal, rhs.GetInt16())).MakeGeneric<bool>(),
                NPTypeCode.UInt16 => NDArray.Scalar(((Func<TLhs, ushort, bool>)func)(lhsVal, rhs.GetUInt16())).MakeGeneric<bool>(),
                NPTypeCode.Int32 => NDArray.Scalar(((Func<TLhs, int, bool>)func)(lhsVal, rhs.GetInt32())).MakeGeneric<bool>(),
                NPTypeCode.UInt32 => NDArray.Scalar(((Func<TLhs, uint, bool>)func)(lhsVal, rhs.GetUInt32())).MakeGeneric<bool>(),
                NPTypeCode.Int64 => NDArray.Scalar(((Func<TLhs, long, bool>)func)(lhsVal, rhs.GetInt64())).MakeGeneric<bool>(),
                NPTypeCode.UInt64 => NDArray.Scalar(((Func<TLhs, ulong, bool>)func)(lhsVal, rhs.GetUInt64())).MakeGeneric<bool>(),
                NPTypeCode.Char => NDArray.Scalar(((Func<TLhs, char, bool>)func)(lhsVal, rhs.GetChar())).MakeGeneric<bool>(),
                NPTypeCode.Single => NDArray.Scalar(((Func<TLhs, float, bool>)func)(lhsVal, rhs.GetSingle())).MakeGeneric<bool>(),
                NPTypeCode.Double => NDArray.Scalar(((Func<TLhs, double, bool>)func)(lhsVal, rhs.GetDouble())).MakeGeneric<bool>(),
                NPTypeCode.Decimal => NDArray.Scalar(((Func<TLhs, decimal, bool>)func)(lhsVal, rhs.GetDecimal())).MakeGeneric<bool>(),
                _ => throw new NotSupportedException($"RHS type {rhsType} not supported")
            };
        }

        /// <summary>
        /// Execute the IL-generated comparison kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ExecuteComparisonKernel(
            ComparisonKernel kernel,
            in NDArray lhs, in NDArray rhs, NDArray<bool> result,
            Shape lhsShape, Shape rhsShape)
        {
            // Get element sizes for offset calculation
            int lhsElemSize = lhs.dtypesize;
            int rhsElemSize = rhs.dtypesize;

            // Calculate base addresses accounting for shape offsets (for sliced views)
            byte* lhsAddr = (byte*)lhs.Address + lhsShape.offset * lhsElemSize;
            byte* rhsAddr = (byte*)rhs.Address + rhsShape.offset * rhsElemSize;

            fixed (int* lhsStrides = lhsShape.strides)
            fixed (int* rhsStrides = rhsShape.strides)
            fixed (int* shape = result.shape)
            {
                kernel(
                    (void*)lhsAddr,
                    (void*)rhsAddr,
                    (bool*)result.Address,
                    lhsStrides,
                    rhsStrides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }

        #region Public API - Comparison Operations (TensorEngine overrides)

        /// <summary>
        /// Element-wise equal comparison (==).
        /// Overrides TensorEngine.Compare - used by the == operator.
        /// </summary>
        public override NDArray<bool> Compare(in NDArray lhs, in NDArray rhs)
            => ExecuteComparisonOp(lhs, rhs, ComparisonOp.Equal);

        /// <summary>
        /// Element-wise not-equal comparison (!=).
        /// </summary>
        public override NDArray<bool> NotEqual(in NDArray lhs, in NDArray rhs)
            => ExecuteComparisonOp(lhs, rhs, ComparisonOp.NotEqual);

        /// <summary>
        /// Element-wise less-than comparison (&lt;).
        /// </summary>
        public override NDArray<bool> Less(in NDArray lhs, in NDArray rhs)
            => ExecuteComparisonOp(lhs, rhs, ComparisonOp.Less);

        /// <summary>
        /// Element-wise less-than-or-equal comparison (&lt;=).
        /// </summary>
        public override NDArray<bool> LessEqual(in NDArray lhs, in NDArray rhs)
            => ExecuteComparisonOp(lhs, rhs, ComparisonOp.LessEqual);

        /// <summary>
        /// Element-wise greater-than comparison (&gt;).
        /// </summary>
        public override NDArray<bool> Greater(in NDArray lhs, in NDArray rhs)
            => ExecuteComparisonOp(lhs, rhs, ComparisonOp.Greater);

        /// <summary>
        /// Element-wise greater-than-or-equal comparison (&gt;=).
        /// </summary>
        public override NDArray<bool> GreaterEqual(in NDArray lhs, in NDArray rhs)
            => ExecuteComparisonOp(lhs, rhs, ComparisonOp.GreaterEqual);

        #endregion
    }
}
