using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Exp(in NDArray nd, Type dtype) => Exp(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise exponential using IL-generated kernels.
        /// </summary>
        public override NDArray Exp(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Exp, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
