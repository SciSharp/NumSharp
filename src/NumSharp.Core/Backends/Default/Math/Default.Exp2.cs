using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Exp2(in NDArray nd, Type dtype) => Exp2(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise 2^x using IL-generated kernels.
        /// </summary>
        public override NDArray Exp2(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Exp2, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
