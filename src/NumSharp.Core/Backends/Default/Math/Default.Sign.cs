using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Sign(in NDArray nd, Type dtype) => Sign(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise sign function using IL-generated kernels.
        /// Returns -1, 0, or 1 based on input sign.
        /// </summary>
        public override NDArray Sign(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Sign, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
