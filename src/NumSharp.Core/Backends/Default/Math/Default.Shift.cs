using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    /// Bit shift operations: left_shift and right_shift.
    /// Integer types only. Uses arithmetic shift for signed types.
    /// SIMD optimized for scalar shift amounts, scalar loop for array shifts.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Bitwise left shift (x1 &lt;&lt; x2).
        /// </summary>
        public override NDArray LeftShift(NDArray lhs, NDArray rhs)
        {
            ValidateIntegerType(lhs, "left_shift");
            ValidateIntegerType(rhs, "left_shift");
            return ExecuteShiftOp(lhs, rhs, isLeftShift: true);
        }

        /// <summary>
        /// Bitwise left shift (x1 &lt;&lt; scalar).
        /// SIMD optimized for contiguous arrays.
        /// </summary>
        public override NDArray LeftShift(NDArray lhs, ValueType rhs)
        {
            ValidateIntegerType(lhs, "left_shift");
            return ExecuteShiftOpScalar(lhs, rhs, isLeftShift: true);
        }

        /// <summary>
        /// Bitwise right shift (x1 &gt;&gt; x2).
        /// Uses arithmetic shift for signed types (sign bit extended).
        /// Uses logical shift for unsigned types (zeros filled).
        /// </summary>
        public override NDArray RightShift(NDArray lhs, NDArray rhs)
        {
            ValidateIntegerType(lhs, "right_shift");
            ValidateIntegerType(rhs, "right_shift");
            return ExecuteShiftOp(lhs, rhs, isLeftShift: false);
        }

        /// <summary>
        /// Bitwise right shift (x1 &gt;&gt; scalar).
        /// SIMD optimized for contiguous arrays.
        /// </summary>
        public override NDArray RightShift(NDArray lhs, ValueType rhs)
        {
            ValidateIntegerType(lhs, "right_shift");
            return ExecuteShiftOpScalar(lhs, rhs, isLeftShift: false);
        }

        /// <summary>
        /// Validate that the array is an integer type.
        /// </summary>
        private static void ValidateIntegerType(NDArray arr, string opName)
        {
            var typeCode = arr.GetTypeCode;
            if (typeCode != NPTypeCode.Byte &&
                typeCode != NPTypeCode.Int16 && typeCode != NPTypeCode.UInt16 &&
                typeCode != NPTypeCode.Int32 && typeCode != NPTypeCode.UInt32 &&
                typeCode != NPTypeCode.Int64 && typeCode != NPTypeCode.UInt64)
            {
                throw new NotSupportedException($"{opName} only supports integer types, got {typeCode}");
            }
        }

        /// <summary>
        /// Execute shift operation with array operands (element-wise shifts).
        /// Uses IL kernel for scalar loop (no SIMD for variable shift amounts).
        /// </summary>
        private unsafe NDArray ExecuteShiftOp(NDArray lhs, NDArray rhs, bool isLeftShift)
        {
            var (broadcastedLhs, broadcastedRhs) = np.broadcast_arrays(lhs, rhs);
            // Create result with clean (contiguous) strides, not broadcast strides
            var resultDimensions = broadcastedLhs.shape;
            var result = new NDArray(lhs.typecode, new Shape(resultDimensions), fillZeros: false);
            var len = result.size;

            // Materialize non-contiguous arrays to allow raw pointer access.
            // This handles broadcast arrays where stride=0 would cause incorrect reads.
            var contiguousLhs = broadcastedLhs.Shape.IsContiguous ? broadcastedLhs : broadcastedLhs.copy();

            // Cast RHS to Int32 for shift amounts (C# shift operators require int for shift amount).
            // Also materialize if non-contiguous to allow raw pointer access.
            var rhsInt32 = broadcastedRhs.GetTypeCode == NPTypeCode.Int32
                ? broadcastedRhs
                : broadcastedRhs.astype(NPTypeCode.Int32);
            var contiguousRhs = rhsInt32.Shape.IsContiguous ? rhsInt32 : rhsInt32.copy();

            var shiftPtr = (int*)contiguousRhs.Address;

            switch (lhs.GetTypeCode)
            {
                case NPTypeCode.Byte:
                    ExecuteShiftArray<byte>(contiguousLhs, shiftPtr, result, len, isLeftShift);
                    break;
                case NPTypeCode.Int16:
                    ExecuteShiftArray<short>(contiguousLhs, shiftPtr, result, len, isLeftShift);
                    break;
                case NPTypeCode.UInt16:
                    ExecuteShiftArray<ushort>(contiguousLhs, shiftPtr, result, len, isLeftShift);
                    break;
                case NPTypeCode.Int32:
                    ExecuteShiftArray<int>(contiguousLhs, shiftPtr, result, len, isLeftShift);
                    break;
                case NPTypeCode.UInt32:
                    ExecuteShiftArray<uint>(contiguousLhs, shiftPtr, result, len, isLeftShift);
                    break;
                case NPTypeCode.Int64:
                    ExecuteShiftArray<long>(contiguousLhs, shiftPtr, result, len, isLeftShift);
                    break;
                case NPTypeCode.UInt64:
                    ExecuteShiftArray<ulong>(contiguousLhs, shiftPtr, result, len, isLeftShift);
                    break;
                default:
                    throw new NotSupportedException($"Shift operations not supported for {lhs.GetTypeCode}");
            }

            return result;
        }

        /// <summary>
        /// Execute element-wise shift using IL kernel.
        /// </summary>
        private static unsafe void ExecuteShiftArray<T>(NDArray input, int* shifts, NDArray output, long count, bool isLeftShift) where T : unmanaged
        {
            var kernel = ILKernelGenerator.GetShiftArrayKernel<T>(isLeftShift);
            if (kernel != null)
            {
                kernel((T*)input.Address, shifts, (T*)output.Address, count);
            }
            else
            {
                // Fallback: scalar loop (should not happen if IL generation is enabled)
                var inPtr = (T*)input.Address;
                var outPtr = (T*)output.Address;
                for (long i = 0; i < count; i++)
                {
                    outPtr[i] = ShiftScalar(inPtr[i], shifts[i], isLeftShift);
                }
            }
        }

        /// <summary>
        /// Execute shift operation with scalar operand (uniform shift).
        /// SIMD optimized path for contiguous arrays.
        /// </summary>
        private unsafe NDArray ExecuteShiftOpScalar(NDArray lhs, ValueType rhs, bool isLeftShift)
        {
            int shiftAmount = Convert.ToInt32(rhs);

            // For contiguous arrays, allocate result and use SIMD kernel
            // For sliced arrays, clone first then apply shift in-place
            NDArray result;
            NDArray input;

            if (lhs.Shape.IsContiguous)
            {
                result = new NDArray(lhs.typecode, new Shape(lhs.shape), fillZeros: false);
                input = lhs;
            }
            else
            {
                result = lhs.Clone();  // Clone also handles non-contiguous arrays
                input = result;        // Shift in-place on the cloned result
            }

            var len = result.size;

            switch (lhs.GetTypeCode)
            {
                case NPTypeCode.Byte:
                    ExecuteShiftScalar<byte>(input, result, shiftAmount, len, isLeftShift);
                    break;
                case NPTypeCode.Int16:
                    ExecuteShiftScalar<short>(input, result, shiftAmount, len, isLeftShift);
                    break;
                case NPTypeCode.UInt16:
                    ExecuteShiftScalar<ushort>(input, result, shiftAmount, len, isLeftShift);
                    break;
                case NPTypeCode.Int32:
                    ExecuteShiftScalar<int>(input, result, shiftAmount, len, isLeftShift);
                    break;
                case NPTypeCode.UInt32:
                    ExecuteShiftScalar<uint>(input, result, shiftAmount, len, isLeftShift);
                    break;
                case NPTypeCode.Int64:
                    ExecuteShiftScalar<long>(input, result, shiftAmount, len, isLeftShift);
                    break;
                case NPTypeCode.UInt64:
                    ExecuteShiftScalar<ulong>(input, result, shiftAmount, len, isLeftShift);
                    break;
                default:
                    throw new NotSupportedException($"Shift operations not supported for {lhs.GetTypeCode}");
            }

            return result;
        }

        /// <summary>
        /// Execute scalar shift using IL kernel (SIMD optimized).
        /// </summary>
        private static unsafe void ExecuteShiftScalar<T>(NDArray input, NDArray output, int shiftAmount, long count, bool isLeftShift) where T : unmanaged
        {
            var kernel = ILKernelGenerator.GetShiftScalarKernel<T>(isLeftShift);
            if (kernel != null)
            {
                kernel((T*)input.Address, (T*)output.Address, shiftAmount, count);
            }
            else
            {
                // Fallback: scalar loop (should not happen if IL generation is enabled)
                var inPtr = (T*)input.Address;
                var outPtr = (T*)output.Address;
                for (long i = 0; i < count; i++)
                {
                    outPtr[i] = ShiftScalar(inPtr[i], shiftAmount, isLeftShift);
                }
            }
        }

        /// <summary>
        /// Fallback scalar shift operation for a single element.
        /// </summary>
        private static T ShiftScalar<T>(T value, int shift, bool isLeftShift) where T : unmanaged
        {
            // Use dynamic to handle all integer types
            // This is only used as fallback when IL kernel is not available
            if (typeof(T) == typeof(byte))
            {
                var v = (byte)(object)value;
                return (T)(object)(byte)(isLeftShift ? (v << shift) : (v >> shift));
            }
            if (typeof(T) == typeof(short))
            {
                var v = (short)(object)value;
                return (T)(object)(short)(isLeftShift ? (v << shift) : (v >> shift));
            }
            if (typeof(T) == typeof(ushort))
            {
                var v = (ushort)(object)value;
                return (T)(object)(ushort)(isLeftShift ? (v << shift) : (v >> shift));
            }
            if (typeof(T) == typeof(int))
            {
                var v = (int)(object)value;
                return (T)(object)(isLeftShift ? (v << shift) : (v >> shift));
            }
            if (typeof(T) == typeof(uint))
            {
                var v = (uint)(object)value;
                return (T)(object)(isLeftShift ? (v << shift) : (v >> shift));
            }
            if (typeof(T) == typeof(long))
            {
                var v = (long)(object)value;
                return (T)(object)(isLeftShift ? (v << shift) : (v >> shift));
            }
            if (typeof(T) == typeof(ulong))
            {
                var v = (ulong)(object)value;
                return (T)(object)(isLeftShift ? (v << shift) : (v >> shift));
            }
            throw new NotSupportedException($"Shift not supported for type {typeof(T)}");
        }
    }
}
