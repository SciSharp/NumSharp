using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Expm1(NDArray nd, Type dtype) => Expm1(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise exp(x) - 1 using IL-generated kernels.
        /// </summary>
        public override NDArray Expm1(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Expm1, ResolveUnaryFloatReturnType(nd, typeCode, "expm1"), @out, where);
        }
    }
}
