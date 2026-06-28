using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cbrt(NDArray nd, Type dtype) => Cbrt(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise cube root using IL-generated kernels.
        /// Computes the cube root of each element.
        /// </summary>
        public override NDArray Cbrt(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Cbrt, ResolveUnaryFloatReturnType(nd, typeCode, "cbrt"), @out, where);
        }
    }
}
