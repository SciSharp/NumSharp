using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Ceil(NDArray nd, Type dtype) => Ceil(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise ceiling using IL-generated kernels.
        /// </summary>
        public override NDArray Ceil(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Ceil, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
