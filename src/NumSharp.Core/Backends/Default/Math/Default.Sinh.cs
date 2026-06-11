using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Sinh(NDArray nd, Type dtype) => Sinh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise hyperbolic sine using IL-generated kernels.
        /// </summary>
        public override NDArray Sinh(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Sinh, ResolveUnaryFloatReturnType(nd, typeCode, "sinh"), @out, where);
        }
    }
}
