using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise cosine using IL-generated kernels.
        /// </summary>
        public override NDArray Cos(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            return ExecuteUnaryOp(nd, UnaryOp.Cos, ResolveUnaryFloatReturnType(nd, typeCode, "cos"), @out, where);
        }
    }
}
