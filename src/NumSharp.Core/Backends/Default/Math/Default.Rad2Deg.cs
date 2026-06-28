using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Rad2Deg(NDArray nd, Type dtype) => Rad2Deg(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise radians to degrees conversion using IL-generated kernels.
        /// Computes x * (180/π).
        /// </summary>
        public override NDArray Rad2Deg(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.Rad2Deg, ResolveUnaryFloatReturnType(nd, typeCode, "rad2deg"), @out, where);
        }
    }
}
