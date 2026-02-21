using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log1p(in NDArray nd, Type dtype) => Log1p(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise log(1 + x) using IL-generated kernels.
        /// </summary>
        public override NDArray Log1p(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Log1p, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
