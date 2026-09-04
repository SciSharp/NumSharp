using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise radians to degrees conversion using IL-generated kernels.
        /// Computes x * (180/π).
        /// </summary>
        public override NDArray Rad2Deg(NDArray nd, Type dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Rad2Deg, ResolveUnaryFloatReturnType(nd, typeCode, "rad2deg"), @out, where);
        }
    }
}
