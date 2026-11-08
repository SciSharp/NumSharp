using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for positive infinity (np.isposinf). Result is a bool array
        /// where True indicates the element is +Inf.
        /// </summary>
        /// <remarks>
        /// NumPy behavior (numpy/lib/_ufunclike_impl.py, <c>isinf(x) &amp; ~signbit(x)</c>):
        /// - Float/Double/Half: True iff value is +Inf (NaN, -Inf, finite -> False)
        /// - Integer/bool types: Always False (integers cannot be Inf)
        /// - Complex: refused here, first (TypeError, NumPy's signbit is ambiguous on complex).
        /// The fused kernel computes <c>x == +inf</c> in one pass (single-pass SIMD for
        /// Single/Double, scalar for Half; cf. NumPy's four-pass isinf/signbit/and).
        /// </remarks>
        public override NDArray IsPosInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy's isposinf / isneginf refuse complex input before anything else: signbit has no complex loop,
            // which NumPy words this way.
            if (a.typecode == NPTypeCode.Complex)
                throw new TypeError(
                    "This operation is not supported for complex128 values because it would be ambiguous.");
            // NumPy computes isposinf/isneginf through isinf (NumPy's functions take no dtype=), so a string input
            // reports isinf's plain refusal.
            Strings.NDStringUfuncs.ThrowIfUnary("isinf", a);
            if (@out is null && where is null)
            {
                using var result = ExecuteUnaryOp(a, UnaryOp.IsPosInf, NPTypeCode.Boolean);
                return result.AsGeneric<bool>();
            }

            return ExecuteUnaryOp(a, UnaryOp.IsPosInf, NPTypeCode.Boolean, @out, where);
        }

        /// <summary>
        /// Test element-wise for negative infinity (np.isneginf). Result is a bool array
        /// where True indicates the element is -Inf.
        /// </summary>
        /// <remarks>
        /// NumPy behavior (numpy/lib/_ufunclike_impl.py, <c>isinf(x) &amp; signbit(x)</c>):
        /// - Float/Double/Half: True iff value is -Inf (NaN, +Inf, finite -> False)
        /// - Integer/bool types: Always False
        /// - Complex: refused here, first (as IsPosInf).
        /// Fused single-pass kernel computing <c>x == -inf</c>.
        /// </remarks>
        public override NDArray IsNegInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy's isposinf / isneginf refuse complex input before anything else: signbit has no complex loop,
            // which NumPy words this way.
            if (a.typecode == NPTypeCode.Complex)
                throw new TypeError(
                    "This operation is not supported for complex128 values because it would be ambiguous.");
            // NumPy computes isposinf/isneginf through isinf (NumPy's functions take no dtype=), so a string input
            // reports isinf's plain refusal.
            Strings.NDStringUfuncs.ThrowIfUnary("isinf", a);
            if (@out is null && where is null)
            {
                using var result = ExecuteUnaryOp(a, UnaryOp.IsNegInf, NPTypeCode.Boolean);
                return result.AsGeneric<bool>();
            }

            return ExecuteUnaryOp(a, UnaryOp.IsNegInf, NPTypeCode.Boolean, @out, where);
        }
    }
}
