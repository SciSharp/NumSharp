using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ATan(NDArray nd, Type dtype) => ATan(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise inverse tangent (arctan) using IL-generated kernels.
        /// </summary>
        public override NDArray ATan(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.ATan, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
