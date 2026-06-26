using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Bit shift operations: left_shift and right_shift.
    ///
    /// NumPy alignment (probed against NumPy 2.4.2): both are integer ufuncs whose loops are
    /// all same-type (<c>bb-&gt;b</c> .. <c>QQ-&gt;Q</c>). Mixed operands therefore promote to
    /// <c>result_type(x1, x2)</c> and the shift runs at that width; bool inputs (no bool loop)
    /// promote to int8. The op is wired into the shared binary pipeline
    /// (<see cref="DefaultEngine.ExecuteBinaryOp"/>) so promotion, broadcasting, strided/sliced
    /// views and scalar×scalar all flow through NpyIter + the IL scalar kernel — the per-element
    /// shift IL lives in <see cref="DirectILKernelGenerator.EmitShiftFromStack"/>. The common
    /// <c>array &lt;&lt; scalar</c> case takes a dedicated 4×-unrolled SIMD kernel
    /// (<see cref="DirectILKernelGenerator.GetShiftScalarKernel{T}"/>).
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Bitwise left shift (x1 &lt;&lt; x2).
        /// </summary>
        public override NDArray LeftShift(NDArray lhs, NDArray rhs)
        {
            ValidateShiftType(lhs, "left_shift");
            ValidateShiftType(rhs, "left_shift");
            return ExecuteShift(lhs, rhs, isLeftShift: true);
        }

        /// <summary>
        /// Bitwise right shift (x1 &gt;&gt; x2).
        /// Arithmetic shift for signed types (sign bit extended); logical shift for unsigned.
        /// </summary>
        public override NDArray RightShift(NDArray lhs, NDArray rhs)
        {
            ValidateShiftType(lhs, "right_shift");
            ValidateShiftType(rhs, "right_shift");
            return ExecuteShift(lhs, rhs, isLeftShift: false);
        }

        /// <summary>
        /// Validate that the array dtype has a shift loop. NumPy's left_shift/right_shift loops
        /// cover bool and the integer dtypes; bool promotes to int8 (handled in
        /// <see cref="ExecuteBinaryOp"/>). Char rides along as a NumSharp integer extension.
        /// Float/complex/decimal raise NumPy's verbatim no-loop TypeError.
        /// </summary>
        private static void ValidateShiftType(NDArray arr, string opName)
        {
            var typeCode = arr.GetTypeCode;
            if (typeCode.IsInteger() || typeCode == NPTypeCode.Boolean || typeCode == NPTypeCode.Char)
                return;

            throw new TypeError($"ufunc '{opName}' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
        }

        /// <summary>
        /// Resolve a shift through the shared binary pipeline. The hot <c>array &lt;&lt; scalar</c>
        /// case is intercepted by the SIMD kernel; everything else (mixed dtype, strided,
        /// broadcast, scalar×scalar) flows through <see cref="ExecuteBinaryOp"/>, which handles
        /// NEP50 promotion and drives the per-element shift IL via NpyIter.
        /// </summary>
        private unsafe NDArray ExecuteShift(NDArray lhs, NDArray rhs, bool isLeftShift)
        {
            var fast = TrySimdScalarShift(lhs, rhs, isLeftShift);
            if (fast is not null)
                return fast;

            var op = isLeftShift ? BinaryOp.LeftShift : BinaryOp.RightShift;
            return ExecuteBinaryOp(lhs, rhs, op);
        }

        /// <summary>
        /// NumPy shift promotion: the same-type loop selected for <c>result_type(lhs, rhs)</c>,
        /// with bool bumped to int8 (no bool shift loop). Mirrors the promotion
        /// <see cref="ExecuteBinaryOp"/> applies, so the SIMD fast path and the general path agree.
        /// </summary>
        private static NPTypeCode ShiftResultType(NDArray lhs, NDArray rhs)
        {
            var rt = np._FindCommonType(lhs, rhs);
            return rt == NPTypeCode.Boolean ? NPTypeCode.SByte : rt;
        }

        /// <summary>
        /// SIMD fast path for <c>contiguous array &lt;&lt; scalar</c>. The shift amount is uniform,
        /// so the overflow check is resolved once and the 4×-unrolled <c>Vector{N}.Shift*</c>
        /// kernel runs over the whole buffer. Returns null (→ <see cref="ExecuteBinaryOp"/>) when
        /// the shape is not array-vs-scalar, the value operand is non-contiguous, or the promoted
        /// dtype has no vector shift (Char).
        /// </summary>
        private unsafe NDArray TrySimdScalarShift(NDArray lhs, NDArray rhs, bool isLeftShift)
        {
            // Only array (value) << scalar (count). scalar×scalar and scalar<<array fall through.
            bool rhsScalar = rhs.Shape.IsScalar || rhs.size == 1;
            bool lhsArray = !(lhs.Shape.IsScalar || lhs.size <= 1);
            if (!rhsScalar || !lhsArray)
                return null;

            var resultType = ShiftResultType(lhs, rhs);
            if (!DirectILKernelGenerator.IsShiftSimdSupported(resultType))
                return null;

            // The kernel walks the value buffer linearly, so the value operand (widened to the
            // result dtype) must be contiguous. A same-dtype strided view defers to NpyIter.
            NDArray value;
            if (lhs.GetTypeCode != resultType)
                value = lhs.astype(resultType);          // contiguous C-order copy at the loop dtype
            else if (lhs.Shape.IsContiguous)
                value = lhs;
            else
                return null;

            if (!value.Shape.IsContiguous)
                return null;

            int bitWidth = resultType.SizeOf() * 8;
            int shiftArg = ReadSaturatedShiftCount(rhs, bitWidth);

            var result = new NDArray(resultType, new Shape((long[])value.shape.Clone()), false);
            if (result.size == 0)
                return result;

            NpFunc.Invoke(resultType, SimdScalarShiftDispatch<int>, value, result, shiftArg, isLeftShift);
            return result;
        }

        /// <summary>
        /// Read the single shift count from a scalar/size-1 operand and saturate it into
        /// <c>[0, bitWidth]</c>: any count that is negative or &gt;= <paramref name="bitWidth"/>
        /// maps to <paramref name="bitWidth"/> so the kernel's once-per-call overflow branch
        /// fires (left/unsigned-right → 0, signed-right → sign fill), matching NumPy. Reading at
        /// the operand's own dtype preserves the magnitude decision regardless of promotion.
        /// </summary>
        private static unsafe int ReadSaturatedShiftCount(NDArray rhs, int bitWidth)
        {
            byte* p = (byte*)rhs.Address + (long)rhs.Shape.offset * rhs.dtypesize;
            long s;
            switch (rhs.GetTypeCode)
            {
                case NPTypeCode.Boolean: s = (*p != 0) ? 1 : 0; break;
                case NPTypeCode.Byte:    s = *p; break;
                case NPTypeCode.SByte:   s = *(sbyte*)p; break;
                case NPTypeCode.Int16:   s = *(short*)p; break;
                case NPTypeCode.UInt16:  s = *(ushort*)p; break;
                case NPTypeCode.Char:    s = *(char*)p; break;
                case NPTypeCode.Int32:   s = *(int*)p; break;
                case NPTypeCode.UInt32:  s = *(uint*)p; break;
                case NPTypeCode.Int64:   s = *(long*)p; break;
                case NPTypeCode.UInt64:
                {
                    ulong u = *(ulong*)p;
                    return u >= (ulong)bitWidth ? bitWidth : (int)u;
                }
                default: return bitWidth;
            }
            return (s < 0 || s >= bitWidth) ? bitWidth : (int)s;
        }

        /// <summary>
        /// Typecode-dispatched (via <see cref="NpFunc"/>) invocation of the SIMD scalar-shift
        /// kernel. The value operand is contiguous; its base address honours
        /// <see cref="Shape.offset"/> so a contiguous slice is handled without a copy.
        /// </summary>
        private static unsafe void SimdScalarShiftDispatch<T>(NDArray value, NDArray output, int shiftArg, bool isLeftShift) where T : unmanaged
        {
            var kernel = DirectILKernelGenerator.GetShiftScalarKernel<T>(isLeftShift);
            if (kernel == null)
                throw new NotSupportedException($"Shift SIMD kernel unavailable for {typeof(T).Name}.");

            byte* inBase = (byte*)value.Address + (long)value.Shape.offset * value.dtypesize;
            kernel((T*)inBase, (T*)output.Address, shiftArg, output.size);
        }
    }
}
