using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise radians to degrees conversion using IL-generated kernels.
        /// Computes x * (180/π).
        /// </summary>
        public override NDArray Rad2Deg(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);

            // rad2deg has no complex loop, so a complex input with no explicit dtype reaches no loop:
            // NumPy raises the generic ufunc TypeError (NOT the kernel's NotSupportedException) and
            // validates the LOOP, not the data — so a zero-size complex operand is rejected too
            // (probed 2.4.2). Same guard shape as np.fabs; resolves the oracle K4/K5 excuses.
            if (typeCode is null && nd.GetTypeCode == NPTypeCode.Complex)
                throw new TypeError(
                    "ufunc 'rad2deg' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            return ExecuteUnaryOp(nd, UnaryOp.Rad2Deg, ResolveUnaryFloatReturnType(nd, typeCode, "rad2deg"), @out, where);
        }
    }
}
