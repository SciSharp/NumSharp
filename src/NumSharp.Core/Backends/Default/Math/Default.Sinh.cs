using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Sinh(in NDArray nd, Type dtype) => Sinh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise hyperbolic sine using IL-generated kernels.
        /// </summary>
        public override NDArray Sinh(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Sinh, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
