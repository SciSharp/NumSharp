using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Bit shift operations: left_shift and right_shift.
    /// Integer types only. Uses arithmetic shift for signed types.
    /// SIMD optimized for scalar shift amounts, scalar loop for array shifts.
    /// </summary>
    public partial class DefaultEngine
    {
        private static unsafe void ShiftArrayDispatch<T>(NDArray input, nint shifts, NDArray result, long len, bool isLeftShift) where T : unmanaged
            => ExecuteShiftArray<T>(input, (int*)shifts, result, len, isLeftShift);

        private static void ShiftScalarDispatch<T>(NDArray input, NDArray result, int shiftAmount, long len, bool isLeftShift) where T : unmanaged
            => ExecuteShiftScalar<T>(input, result, shiftAmount, len, isLeftShift);

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
        /// Validate that the array is an integer type.
        /// Raises TypeError to match NumPy's ufunc dtype rejection.
        /// </summary>
        private static void ValidateIntegerType(NDArray arr, string opName)
        {
            var typeCode = arr.GetTypeCode;
            if (typeCode != NPTypeCode.Byte && typeCode != NPTypeCode.SByte &&
                typeCode != NPTypeCode.Int16 && typeCode != NPTypeCode.UInt16 &&
                typeCode != NPTypeCode.Int32 && typeCode != NPTypeCode.UInt32 &&
                typeCode != NPTypeCode.Int64 && typeCode != NPTypeCode.UInt64)
            {
                throw new TypeError($"ufunc '{opName}' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
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

            // Shift amounts must be Int32 (C# shift operators require an int amount). astype
            // performs the genuine dtype conversion when needed; an already-Int32 operand keeps
            // its (possibly strided / broadcast) view and is read through its strides below.
            var rhsInt32 = broadcastedRhs.GetTypeCode == NPTypeCode.Int32
                ? broadcastedRhs
                : broadcastedRhs.astype(NPTypeCode.Int32);

            bool lhsFlat = broadcastedLhs.Shape.IsContiguous && broadcastedLhs.Shape.offset == 0;
            bool rhsFlat = rhsInt32.Shape.IsContiguous && rhsInt32.Shape.offset == 0;
            if (lhsFlat && rhsFlat)
            {
                // Fast path: both operands plainly contiguous — IL scalar-loop kernel.
                var shiftPtr = (int*)rhsInt32.Address;
                NpFunc.Invoke(lhs.GetTypeCode, ShiftArrayDispatch<int>, broadcastedLhs, (nint)shiftPtr, result, len, isLeftShift);
            }
            else
            {
                // Strided / sliced / broadcast (stride=0) operands: walk both inputs in C-order
                // through their own strides — no materializing copy (the rejected anti-pattern).
                ExecuteShiftArrayStrided(broadcastedLhs, rhsInt32, result, len, isLeftShift);
            }

            return result;
        }

        /// <summary>
        /// Element-wise shift over operands of arbitrary layout. Both the values and the per-element
        /// shift amounts are read through their own strides (<see cref="FlatStrideOffset"/>), so
        /// strided / transposed / sliced / broadcast (stride=0) views are consumed in place; the
        /// result is freshly C-contiguous. Matches the contiguous IL kernel element-for-element.
        /// </summary>
        private static unsafe void ExecuteShiftArrayStrided(NDArray input, NDArray shifts, NDArray output, long count, bool isLeftShift)
        {
            var dims = input.shape;
            int ndim = input.ndim;
            var inStr = input.strides;
            var shStr = shifts.strides;
            byte* inBase = (byte*)input.Address + input.Shape.offset * input.dtypesize;
            int* shPtr = (int*)((byte*)shifts.Address + shifts.Shape.offset * shifts.dtypesize);
            bool inC = input.Shape.IsContiguous && input.Shape.offset == 0;
            bool shC = shifts.Shape.IsContiguous && shifts.Shape.offset == 0;

            switch (input.GetTypeCode)
            {
                case NPTypeCode.Byte:
                {
                    var s = (byte*)inBase; var o = (byte*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = (byte)(isLeftShift ? (v << sh) : (v >> sh)); }
                    break;
                }
                case NPTypeCode.SByte:
                {
                    var s = (sbyte*)inBase; var o = (sbyte*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = (sbyte)(isLeftShift ? (v << sh) : (v >> sh)); }
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var s = (short*)inBase; var o = (short*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = (short)(isLeftShift ? (v << sh) : (v >> sh)); }
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var s = (ushort*)inBase; var o = (ushort*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = (ushort)(isLeftShift ? (v << sh) : (v >> sh)); }
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var s = (int*)inBase; var o = (int*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = isLeftShift ? (v << sh) : (v >> sh); }
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var s = (uint*)inBase; var o = (uint*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = isLeftShift ? (v << sh) : (v >> sh); }
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var s = (long*)inBase; var o = (long*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = isLeftShift ? (v << sh) : (v >> sh); }
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var s = (ulong*)inBase; var o = (ulong*)output.Address;
                    for (long i = 0; i < count; i++) { var v = s[inC ? i : FlatStrideOffset(i, dims, inStr, ndim)]; int sh = shPtr[shC ? i : FlatStrideOffset(i, dims, shStr, ndim)]; o[i] = isLeftShift ? (v << sh) : (v >> sh); }
                    break;
                }
                default:
                    throw new NotSupportedException($"Shift not supported for type {input.GetTypeCode}");
            }
        }

        /// <summary>
        /// Execute element-wise shift using IL kernel.
        /// </summary>
        private static unsafe void ExecuteShiftArray<T>(NDArray input, int* shifts, NDArray output, long count, bool isLeftShift) where T : unmanaged
        {
            var kernel = DirectILKernelGenerator.GetShiftArrayKernel<T>(isLeftShift);
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
        /// Execute scalar shift using IL kernel (SIMD optimized).
        /// </summary>
        private static unsafe void ExecuteShiftScalar<T>(NDArray input, NDArray output, int shiftAmount, long count, bool isLeftShift) where T : unmanaged
        {
            var kernel = DirectILKernelGenerator.GetShiftScalarKernel<T>(isLeftShift);
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
            if (typeof(T) == typeof(sbyte))
            {
                var v = (sbyte)(object)value;
                return (T)(object)(sbyte)(isLeftShift ? (v << shift) : (v >> shift));
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
