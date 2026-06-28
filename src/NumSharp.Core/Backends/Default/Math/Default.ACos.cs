using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ACos(NDArray nd, Type dtype) => ACos(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise inverse cosine (arccos) using IL-generated kernels.
        /// </summary>
        public override NDArray ACos(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: the where bool check is argument
            // parsing -- it precedes loop resolution (the dtype= no-loop
            // raise inside ResolveUnaryFloatReturnType).
            ValidateWhereMask(where);
            return ExecuteUnaryOp(nd, UnaryOp.ACos, ResolveUnaryFloatReturnType(nd, typeCode, "arccos"), @out, where);
        }
    }
}
