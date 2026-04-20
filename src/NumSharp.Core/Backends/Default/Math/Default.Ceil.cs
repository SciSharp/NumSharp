using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Ceil(NDArray nd, Type dtype) => Ceil(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise ceiling using IL-generated kernels.
        /// NumPy: for integer dtypes, ceil is a no-op that preserves the input dtype.
        /// </summary>
        public override NDArray Ceil(NDArray nd, NPTypeCode? typeCode = null)
        {
            if (!typeCode.HasValue && nd.GetTypeCode.IsInteger())
                return Cast(nd, nd.GetTypeCode, copy: true);
            return ExecuteUnaryOp(nd, UnaryOp.Ceil, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
