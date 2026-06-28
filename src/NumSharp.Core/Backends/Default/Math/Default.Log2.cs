using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log2(NDArray nd, Type dtype) => Log2(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise log base 2 using IL-generated kernels.
        /// </summary>
        public override NDArray Log2(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Log2, ResolveUnaryFloatReturnType(nd, typeCode, "log2"), @out, where);
        }
    }
}
