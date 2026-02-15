using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Tan(in NDArray nd, Type dtype) => Tan(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise tangent using IL-generated kernels.
        /// </summary>
        public override NDArray Tan(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Tan, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
