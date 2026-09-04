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
        /// - Complex: rejected upstream at the np.* layer (TypeError, NumPy's signbit is
        ///   ambiguous on complex), so the engine predicate never sees Complex.
        /// The fused kernel computes <c>x == +inf</c> in one pass (single-pass SIMD for
        /// Single/Double, scalar for Half; cf. NumPy's four-pass isinf/signbit/and).
        /// </remarks>
        public override NDArray IsPosInf(NDArray a, Type dtype = null, NDArray @out = null, NDArray where = null)
        {
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
        /// - Complex: rejected upstream at the np.* layer.
        /// Fused single-pass kernel computing <c>x == -inf</c>.
        /// </remarks>
        public override NDArray IsNegInf(NDArray a, Type dtype = null, NDArray @out = null, NDArray where = null)
        {
            if (@out is null && where is null)
            {
                using var result = ExecuteUnaryOp(a, UnaryOp.IsNegInf, NPTypeCode.Boolean);
                return result.AsGeneric<bool>();
            }

            return ExecuteUnaryOp(a, UnaryOp.IsNegInf, NPTypeCode.Boolean, @out, where);
        }
    }
}
