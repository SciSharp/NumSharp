using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Tanh(NDArray nd, Type dtype) => Tanh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise hyperbolic tangent using IL-generated kernels.
        /// </summary>
        public override NDArray Tanh(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Tanh, ResolveUnaryFloatReturnType(nd, typeCode, "tanh"), @out, where);
        }
    }
}
