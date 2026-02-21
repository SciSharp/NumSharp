using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cosh(in NDArray nd, Type dtype) => Cosh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise hyperbolic cosine using IL-generated kernels.
        /// </summary>
        public override NDArray Cosh(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Cosh, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
