using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ACosh(NDArray nd, Type dtype) => ACosh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise inverse hyperbolic cosine (arccosh) using IL-generated kernels.
        /// </summary>
        public override NDArray ACosh(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Acosh, ResolveUnaryFloatReturnType(nd, typeCode, "arccosh"), @out, where);
        }
    }
}
