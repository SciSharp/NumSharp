using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise natural logarithm using IL-generated kernels.
        /// </summary>
        public override NDArray Log(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            return ExecuteUnaryOp(nd, UnaryOp.Log, ResolveUnaryFloatReturnType(nd, typeCode, "log"), @out, where);
        }
    }
}
