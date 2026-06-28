using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cosh(NDArray nd, Type dtype) => Cosh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise hyperbolic cosine using IL-generated kernels.
        /// </summary>
        public override NDArray Cosh(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Cosh, ResolveUnaryFloatReturnType(nd, typeCode, "cosh"), @out, where);
        }
    }
}
