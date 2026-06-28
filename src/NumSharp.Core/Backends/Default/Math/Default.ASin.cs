using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ASin(NDArray nd, Type dtype) => ASin(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise inverse sine (arcsin) using IL-generated kernels.
        /// </summary>
        public override NDArray ASin(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.ASin, ResolveUnaryFloatReturnType(nd, typeCode, "arcsin"), @out, where);
        }
    }
}
