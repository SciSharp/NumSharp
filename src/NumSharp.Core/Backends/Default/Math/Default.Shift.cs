using System;

namespace NumSharp.Backends
{
    /// <summary>
    /// Bit shift operations: left_shift and right_shift.
    /// Integer types only. Uses arithmetic shift for signed types.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Bitwise left shift (x1 &lt;&lt; x2).
        /// </summary>
        public override NDArray LeftShift(in NDArray lhs, in NDArray rhs)
        {
            ValidateIntegerType(lhs, "left_shift");
            ValidateIntegerType(rhs, "left_shift");
            return ExecuteShiftOp(lhs, rhs, isLeftShift: true);
        }

        /// <summary>
        /// Bitwise left shift (x1 &lt;&lt; scalar).
        /// </summary>
        public override NDArray LeftShift(in NDArray lhs, in ValueType rhs)
        {
            ValidateIntegerType(lhs, "left_shift");
            return ExecuteShiftOpScalar(lhs, rhs, isLeftShift: true);
        }

        /// <summary>
        /// Bitwise right shift (x1 &gt;&gt; x2).
        /// Uses arithmetic shift for signed types (sign bit extended).
        /// Uses logical shift for unsigned types (zeros filled).
        /// </summary>
        public override NDArray RightShift(in NDArray lhs, in NDArray rhs)
        {
            ValidateIntegerType(lhs, "right_shift");
            ValidateIntegerType(rhs, "right_shift");
            return ExecuteShiftOp(lhs, rhs, isLeftShift: false);
        }

        /// <summary>
        /// Bitwise right shift (x1 &gt;&gt; scalar).
        /// </summary>
        public override NDArray RightShift(in NDArray lhs, in ValueType rhs)
        {
            ValidateIntegerType(lhs, "right_shift");
            return ExecuteShiftOpScalar(lhs, rhs, isLeftShift: false);
        }

        /// <summary>
        /// Validate that the array is an integer type.
        /// </summary>
        private static void ValidateIntegerType(in NDArray arr, string opName)
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
        /// Execute shift operation with array operands.
        /// </summary>
        private unsafe NDArray ExecuteShiftOp(in NDArray lhs, in NDArray rhs, bool isLeftShift)
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
                {
                    var lhsPtr = (byte*)contiguousLhs.Address;
                    var resPtr = (byte*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            resPtr[i] = (byte)(lhsPtr[i] << shiftPtr[i]);
                    else
                        for (int i = 0; i < len; i++)
                            resPtr[i] = (byte)(lhsPtr[i] >> shiftPtr[i]);
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var lhsPtr = (short*)contiguousLhs.Address;
                    var resPtr = (short*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            resPtr[i] = (short)(lhsPtr[i] << shiftPtr[i]);
                    else
                        for (int i = 0; i < len; i++)
                            resPtr[i] = (short)(lhsPtr[i] >> shiftPtr[i]);
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var lhsPtr = (ushort*)contiguousLhs.Address;
                    var resPtr = (ushort*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            resPtr[i] = (ushort)(lhsPtr[i] << shiftPtr[i]);
                    else
                        for (int i = 0; i < len; i++)
                            resPtr[i] = (ushort)(lhsPtr[i] >> shiftPtr[i]);
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var lhsPtr = (int*)contiguousLhs.Address;
                    var resPtr = (int*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] << shiftPtr[i];
                    else
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] >> shiftPtr[i];
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var lhsPtr = (uint*)contiguousLhs.Address;
                    var resPtr = (uint*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] << shiftPtr[i];
                    else
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] >> shiftPtr[i];
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var lhsPtr = (long*)contiguousLhs.Address;
                    var resPtr = (long*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] << shiftPtr[i];
                    else
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] >> shiftPtr[i];
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var lhsPtr = (ulong*)contiguousLhs.Address;
                    var resPtr = (ulong*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] << shiftPtr[i];
                    else
                        for (int i = 0; i < len; i++)
                            resPtr[i] = lhsPtr[i] >> shiftPtr[i];
                    break;
                }
                default:
                    throw new NotSupportedException($"Shift operations not supported for {lhs.GetTypeCode}");
            }

            return result;
        }

        /// <summary>
        /// Execute shift operation with scalar operand.
        /// </summary>
        private unsafe NDArray ExecuteShiftOpScalar(in NDArray lhs, in ValueType rhs, bool isLeftShift)
        {
            var result = lhs.Clone();
            var len = result.size;
            int shiftAmount = Convert.ToInt32(rhs);

            switch (lhs.GetTypeCode)
            {
                case NPTypeCode.Byte:
                {
                    var ptr = (byte*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            ptr[i] = (byte)(ptr[i] << shiftAmount);
                    else
                        for (int i = 0; i < len; i++)
                            ptr[i] = (byte)(ptr[i] >> shiftAmount);
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var ptr = (short*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            ptr[i] = (short)(ptr[i] << shiftAmount);
                    else
                        for (int i = 0; i < len; i++)
                            ptr[i] = (short)(ptr[i] >> shiftAmount);
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var ptr = (ushort*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            ptr[i] = (ushort)(ptr[i] << shiftAmount);
                    else
                        for (int i = 0; i < len; i++)
                            ptr[i] = (ushort)(ptr[i] >> shiftAmount);
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var ptr = (int*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] << shiftAmount;
                    else
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] >> shiftAmount;
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var ptr = (uint*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] << shiftAmount;
                    else
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] >> shiftAmount;
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var ptr = (long*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] << shiftAmount;
                    else
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] >> shiftAmount;
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var ptr = (ulong*)result.Address;
                    if (isLeftShift)
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] << shiftAmount;
                    else
                        for (int i = 0; i < len; i++)
                            ptr[i] = ptr[i] >> shiftAmount;
                    break;
                }
                default:
                    throw new NotSupportedException($"Shift operations not supported for {lhs.GetTypeCode}");
            }

            return result;
        }
    }
}
