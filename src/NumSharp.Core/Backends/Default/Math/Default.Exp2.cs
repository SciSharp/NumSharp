using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise 2^x using IL-generated kernels.
        /// </summary>
        public override NDArray Exp2(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Exp2, ResolveUnaryFloatReturnType(nd, typeCode, "exp2"), @out, where);
        }
    }
}
