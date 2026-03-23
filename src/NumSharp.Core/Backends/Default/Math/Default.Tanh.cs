using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Tanh(NDArray nd, Type dtype) => Tanh(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise hyperbolic tangent using IL-generated kernels.
        /// </summary>
        public override NDArray Tanh(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Tanh, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
