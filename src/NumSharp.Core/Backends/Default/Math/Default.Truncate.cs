using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Truncate(in NDArray nd, Type dtype) => Truncate(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise truncation (toward zero) using IL-generated kernels.
        /// </summary>
        public override NDArray Truncate(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Truncate, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
