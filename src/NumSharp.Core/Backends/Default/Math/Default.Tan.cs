using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise tangent using IL-generated kernels.
        /// </summary>
        public override NDArray Tan(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            return ExecuteUnaryOp(nd, UnaryOp.Tan, ResolveUnaryFloatReturnType(nd, typeCode, "tan"), @out, where);
        }
    }
}
