using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Tan(NDArray nd, Type dtype) => Tan(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise tangent using IL-generated kernels.
        /// </summary>
        public override NDArray Tan(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Tan, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
