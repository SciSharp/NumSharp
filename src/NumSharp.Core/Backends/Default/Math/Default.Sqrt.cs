using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise square root using IL-generated kernels.
        /// </summary>
        public override NDArray Sqrt(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            return ExecuteUnaryOp(nd, UnaryOp.Sqrt, ResolveUnaryFloatReturnType(nd, typeCode, "sqrt"), @out, where);
        }
    }
}
