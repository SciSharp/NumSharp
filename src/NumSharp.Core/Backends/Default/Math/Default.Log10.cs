using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise log base 10 using IL-generated kernels.
        /// </summary>
        public override NDArray Log10(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Log10, ResolveUnaryFloatReturnType(nd, typeCode, "log10"), @out, where);
        }
    }
}
