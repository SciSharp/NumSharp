using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Sqrt(in NDArray nd, Type dtype) => Sqrt(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise square root using IL-generated kernels.
        /// </summary>
        public override NDArray Sqrt(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Sqrt, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
