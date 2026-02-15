using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Ceil(in NDArray nd, Type dtype) => Ceil(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise ceiling using IL-generated kernels.
        /// </summary>
        public override NDArray Ceil(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Ceil, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
