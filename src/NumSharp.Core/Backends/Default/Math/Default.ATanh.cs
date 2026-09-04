using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise inverse hyperbolic tangent (arctanh) using IL-generated kernels.
        /// </summary>
        public override NDArray ATanh(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Atanh, ResolveUnaryFloatReturnType(nd, typeCode, "arctanh"), @out, where);
        }
    }
}
