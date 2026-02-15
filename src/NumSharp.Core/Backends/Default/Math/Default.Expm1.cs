using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Expm1(in NDArray nd, Type dtype) => Expm1(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise exp(x) - 1 using IL-generated kernels.
        /// </summary>
        public override NDArray Expm1(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Expm1, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
