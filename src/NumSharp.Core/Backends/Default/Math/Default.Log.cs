using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log(in NDArray nd, Type dtype) => Log(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise natural logarithm using IL-generated kernels.
        /// </summary>
        public override NDArray Log(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Log, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
