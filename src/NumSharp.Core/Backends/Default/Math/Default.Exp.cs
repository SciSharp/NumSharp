using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise exponential using IL-generated kernels.
        /// </summary>
        public override NDArray Exp(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            return ExecuteUnaryOp(nd, UnaryOp.Exp, ResolveUnaryFloatReturnType(nd, typeCode, "exp"), @out, where);
        }
    }
}
