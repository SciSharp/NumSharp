using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise degrees to radians conversion using IL-generated kernels.
        /// Computes x * (π/180).
        /// </summary>
        public override NDArray Deg2Rad(NDArray nd, Type dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Deg2Rad, ResolveUnaryFloatReturnType(nd, typeCode, "deg2rad"), @out, where);
        }
    }
}
