using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Truncate(NDArray nd, Type dtype) => Truncate(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise truncation (toward zero) using IL-generated kernels.
        /// NumPy: for integer dtypes, trunc is a no-op that preserves the input dtype.
        /// </summary>
        public override NDArray Truncate(NDArray nd, NPTypeCode? typeCode = null)
        {
            if (!typeCode.HasValue && nd.GetTypeCode.IsInteger())
                return Cast(nd, nd.GetTypeCode, copy: true);
            return ExecuteUnaryOp(nd, UnaryOp.Truncate, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
