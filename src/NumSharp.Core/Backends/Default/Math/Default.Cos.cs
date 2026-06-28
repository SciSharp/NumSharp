using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cos(NDArray nd, Type dtype) => Cos(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise cosine using IL-generated kernels.
        /// </summary>
        public override NDArray Cos(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Cos, ResolveUnaryFloatReturnType(nd, typeCode, "cos"), @out, where);
        }
    }
}
