using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cos(in NDArray nd, Type dtype) => Cos(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise cosine using IL-generated kernels.
        /// </summary>
        public override NDArray Cos(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Cos, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
