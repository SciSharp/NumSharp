using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise inverse hyperbolic sine (arcsinh) using IL-generated kernels.
        /// </summary>
        public override NDArray ASinh(NDArray nd, Type dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Asinh, ResolveUnaryFloatReturnType(nd, typeCode, "arcsinh"), @out, where);
        }
    }
}
