using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise sine using IL-generated kernels.
        /// </summary>
        public override NDArray Sin(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            return ExecuteUnaryOp(nd, UnaryOp.Sin, ResolveUnaryFloatReturnType(nd, typeCode, "sin"), @out, where);
        }
    }
}
