using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Tanh(in NDArray nd, Type dtype) => Tanh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise hyperbolic tangent using IL-generated kernels.
        /// </summary>
        public override NDArray Tanh(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Tanh, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
