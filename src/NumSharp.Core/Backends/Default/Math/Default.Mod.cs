using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise modulo using IL-generated kernels.
        /// Supports all 144 type combinations with automatic type promotion.
        /// </summary>
        public override NDArray Mod(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();

            // mod (NumPy ufunc name 'remainder') has no complex loop, so ANY complex operand with no
            // explicit dtype reaches no loop: NumPy raises the generic ufunc TypeError (NOT the kernel's
            // NotSupportedException) and validates the LOOP, not the data — so a zero-size complex operand
            // is rejected too (probed 2.4.2). Same guard shape as np.fabs; resolves oracle K4/K5.
            if (typeCode is null && (lhs.GetTypeCode == NPTypeCode.Complex || rhs.GetTypeCode == NPTypeCode.Complex))
                throw new TypeError(
                    "ufunc 'remainder' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Mod, @out, where, typeCode);
        }
    }
}
