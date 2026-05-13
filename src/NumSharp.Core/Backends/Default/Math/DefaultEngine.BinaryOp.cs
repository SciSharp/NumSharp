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
        internal unsafe NDArray ExecuteBinaryOp(NDArray lhs, NDArray rhs, BinaryOp op)
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

            // NumPy Power promotion (NEP50). Most cases are already handled by
            // _FindCommonType (which applies weak/strict scalar rules correctly):
            //   - i32_arr ** i32_arr  → int32
            //   - i32_arr ** i64_arr  → int64
            //   - f32_arr ** f64_arr  → float64
            //   - f32_arr ** i32_arr  → float64 (NEP50 strict)
            //   - f32_arr ** Python int (0-D weak) → float32 (NEP50 weak)
            //   - i32_arr ** f32_scalar (cross-array)  → float64 (handled below)
            //
            // The one rule _FindCommonType doesn't cover is `int_scalar ** float_arr`:
            // a 0-D int scalar is treated as "weak", which preserves the float's dtype,
            // but NumPy's int-base + float-exp rule promotes unconditionally to float64.
            // (Group 0=Byte/Char, 1=signed int, 2=unsigned int, 3=float, 4=decimal)
            //
            // Known limitation: explicit 0-D integer arrays (`np.array(2, int32)`)
            // are indistinguishable from C# `int 2` after `np.asanyarray`. NumPy would
            // strict-promote `f32_arr ** np.int32(2)` to float64; NumSharp preserves
            // float32 for both `f32_arr ** 2` (correct) and `f32_arr ** np.array(2, int32)`
            // (misaligned with NumPy but rare in idiomatic C# code).
            if (op == BinaryOp.Power)
            {
                var lhsGroup = lhsType.GetGroup();
                var rhsGroup = rhsType.GetGroup();

                if (lhsGroup <= 2 && rhsGroup == 3)
                {
                    resultType = NPTypeCode.Double;
                }
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
            fixed (long* lhsStrides = leftShape.strides)
            fixed (long* rhsStrides = rightShape.strides)
            fixed (long* shape = resultShape.dimensions)
            {
                path = ClassifyPath(lhsStrides, rhsStrides, shape, resultShape.NDim, resultType);
            }

            // Get kernel key
            var key = new MixedTypeKernelKey(lhsType, rhsType, resultType, op, path);

            // Get or generate kernel
            var kernel = ILKernelGenerator.GetMixedTypeKernel(key);

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

            // NumPy-aligned layout preservation: element-wise ops preserve F-contig
            // when every non-scalar operand is strictly F-contig.
            // Kernels write in linear C-order, so we relay out via copy('F') when needed.
            if (ShouldProduceFContigOutput(lhs, rhs, result.Shape))
                return result.copy('F');

            return result;
        }

        /// <summary>
        /// NumPy-aligned rule: the output is F-contiguous when every non-scalar operand
        /// is strictly F-contiguous (IsFContiguous && !IsContiguous).
        /// Scalars (and 1-element shapes, both C and F) do not change the decision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ShouldProduceFContigOutput(NDArray a, Shape resultShape)
        {
            if (resultShape.NDim <= 1 || resultShape.size <= 1)
                return false;
            var s = a.Shape;
            // Scalars and size-1 shapes don't force a preference.
            if (s.IsScalar || s.size <= 1)
                return false;
            return s.IsFContiguous && !s.IsContiguous;
        }

        /// <summary>
        /// Binary variant — require that every non-scalar operand is strictly F-contiguous
        /// and at least one of them is (otherwise the scalar+scalar case is excluded upstream).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ShouldProduceFContigOutput(NDArray lhs, NDArray rhs, Shape resultShape)
        {
            if (resultShape.NDim <= 1 || resultShape.size <= 1)
                return false;

            bool lhsScalar = lhs.Shape.IsScalar || lhs.Shape.size <= 1;
            bool rhsScalar = rhs.Shape.IsScalar || rhs.Shape.size <= 1;

            bool lhsPureF = !lhsScalar && lhs.Shape.IsFContiguous && !lhs.Shape.IsContiguous;
            bool rhsPureF = !rhsScalar && rhs.Shape.IsFContiguous && !rhs.Shape.IsContiguous;
            bool lhsPureC = !lhsScalar && lhs.Shape.IsContiguous && !lhs.Shape.IsFContiguous;
            bool rhsPureC = !rhsScalar && rhs.Shape.IsContiguous && !rhs.Shape.IsFContiguous;

            // If any non-scalar operand is strictly C-contig, fall through to the C default.
            if (lhsPureC || rhsPureC)
                return false;

            // At least one non-scalar operand must be strictly F-contig to trigger F output.
            return lhsPureF || rhsPureF;
        }

        /// <summary>
        /// Execute scalar × scalar operation using IL-generated delegate.
        /// </summary>
        private NDArray ExecuteScalarScalar(NDArray lhs, NDArray rhs, BinaryOp op, NPTypeCode resultType)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;
            var key = new BinaryScalarKernelKey(lhsType, rhsType, resultType, op);
            var func = ILKernelGenerator.GetBinaryScalarDelegate(key);

            // Dispatch based on lhs type first
            return lhsType switch
            {
                NPTypeCode.Boolean => InvokeBinaryScalarLhs(func, lhs.GetBoolean(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Byte => InvokeBinaryScalarLhs(func, lhs.GetByte(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.SByte => InvokeBinaryScalarLhs(func, lhs.GetSByte(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Int16 => InvokeBinaryScalarLhs(func, lhs.GetInt16(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.UInt16 => InvokeBinaryScalarLhs(func, lhs.GetUInt16(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Int32 => InvokeBinaryScalarLhs(func, lhs.GetInt32(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.UInt32 => InvokeBinaryScalarLhs(func, lhs.GetUInt32(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Int64 => InvokeBinaryScalarLhs(func, lhs.GetInt64(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.UInt64 => InvokeBinaryScalarLhs(func, lhs.GetUInt64(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Char => InvokeBinaryScalarLhs(func, lhs.GetChar(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Half => InvokeBinaryScalarLhs(func, lhs.GetHalf(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Single => InvokeBinaryScalarLhs(func, lhs.GetSingle(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Double => InvokeBinaryScalarLhs(func, lhs.GetDouble(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Decimal => InvokeBinaryScalarLhs(func, lhs.GetDecimal(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Complex => InvokeBinaryScalarLhs(func, lhs.GetComplex(Array.Empty<long>()), rhs, rhsType, resultType),
                _ => throw new NotSupportedException($"LHS type {lhsType} not supported")
            };
        }

        /// <summary>
        /// Continue binary scalar dispatch with typed LHS value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray InvokeBinaryScalarLhs<TLhs>(
            Delegate func, TLhs lhsVal, NDArray rhs, NPTypeCode rhsType, NPTypeCode resultType)
        {
            // Dispatch based on rhs type
            return rhsType switch
            {
                NPTypeCode.Boolean => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetBoolean(Array.Empty<long>()), resultType),
                NPTypeCode.Byte => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetByte(Array.Empty<long>()), resultType),
                NPTypeCode.SByte => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetSByte(Array.Empty<long>()), resultType),
                NPTypeCode.Int16 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt16(Array.Empty<long>()), resultType),
                NPTypeCode.UInt16 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt16(Array.Empty<long>()), resultType),
                NPTypeCode.Int32 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt32(Array.Empty<long>()), resultType),
                NPTypeCode.UInt32 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt32(Array.Empty<long>()), resultType),
                NPTypeCode.Int64 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt64(Array.Empty<long>()), resultType),
                NPTypeCode.UInt64 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt64(Array.Empty<long>()), resultType),
                NPTypeCode.Char => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetChar(Array.Empty<long>()), resultType),
                NPTypeCode.Half => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetHalf(Array.Empty<long>()), resultType),
                NPTypeCode.Single => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetSingle(Array.Empty<long>()), resultType),
                NPTypeCode.Double => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetDouble(Array.Empty<long>()), resultType),
                NPTypeCode.Decimal => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetDecimal(Array.Empty<long>()), resultType),
                NPTypeCode.Complex => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetComplex(Array.Empty<long>()), resultType),
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
                NPTypeCode.SByte => NDArray.Scalar(((Func<TLhs, TRhs, sbyte>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int16 => NDArray.Scalar(((Func<TLhs, TRhs, short>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt16 => NDArray.Scalar(((Func<TLhs, TRhs, ushort>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int32 => NDArray.Scalar(((Func<TLhs, TRhs, int>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt32 => NDArray.Scalar(((Func<TLhs, TRhs, uint>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int64 => NDArray.Scalar(((Func<TLhs, TRhs, long>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt64 => NDArray.Scalar(((Func<TLhs, TRhs, ulong>)func)(lhsVal, rhsVal)),
                NPTypeCode.Char => NDArray.Scalar(((Func<TLhs, TRhs, char>)func)(lhsVal, rhsVal)),
                NPTypeCode.Half => NDArray.Scalar(((Func<TLhs, TRhs, Half>)func)(lhsVal, rhsVal)),
                NPTypeCode.Single => NDArray.Scalar(((Func<TLhs, TRhs, float>)func)(lhsVal, rhsVal)),
                NPTypeCode.Double => NDArray.Scalar(((Func<TLhs, TRhs, double>)func)(lhsVal, rhsVal)),
                NPTypeCode.Decimal => NDArray.Scalar(((Func<TLhs, TRhs, decimal>)func)(lhsVal, rhsVal)),
                NPTypeCode.Complex => NDArray.Scalar(((Func<TLhs, TRhs, System.Numerics.Complex>)func)(lhsVal, rhsVal)),
                _ => throw new NotSupportedException($"Result type {resultType} not supported")
            };
        }

        /// <summary>
        /// Classify execution path based on strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ExecutionPath ClassifyPath(
            long* lhsStrides, long* rhsStrides, long* shape, int ndim, NPTypeCode resultType)
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
            long lhsInner = lhsStrides[ndim - 1];
            long rhsInner = rhsStrides[ndim - 1];
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
            NDArray lhs, NDArray rhs, NDArray result,
            Shape lhsShape, Shape rhsShape)
        {
            // Get element sizes for offset calculation
            int lhsElemSize = lhs.dtypesize;
            int rhsElemSize = rhs.dtypesize;

            // Calculate base addresses accounting for shape offsets (for sliced views)
            // The Shape.offset represents the element offset into the underlying storage
            byte* lhsAddr = (byte*)lhs.Address + lhsShape.offset * lhsElemSize;
            byte* rhsAddr = (byte*)rhs.Address + rhsShape.offset * rhsElemSize;

            fixed (long* lhsStrides = lhsShape.strides)
            fixed (long* rhsStrides = rhsShape.strides)
            fixed (long* shape = result.shape)
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
            NDArray lhs, NDArray rhs, NDArray result,
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
