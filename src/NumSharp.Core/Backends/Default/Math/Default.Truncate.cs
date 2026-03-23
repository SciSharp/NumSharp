using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Truncate(NDArray nd, Type dtype) => Truncate(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise truncation (toward zero) using IL-generated kernels.
        /// </summary>
        public override NDArray Truncate(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Truncate, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
