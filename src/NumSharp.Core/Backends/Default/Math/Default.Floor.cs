using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Floor(NDArray nd, Type dtype) => Floor(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise floor using IL-generated kernels.
        /// NumPy: for integer dtypes, floor is a no-op that preserves the input dtype.
        /// </summary>
        public override NDArray Floor(NDArray nd, NPTypeCode? typeCode = null)
        {
            if (!typeCode.HasValue && nd.GetTypeCode.IsInteger())
                return Cast(nd, nd.GetTypeCode, copy: true);
            return ExecuteUnaryOp(nd, UnaryOp.Floor, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
