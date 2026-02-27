using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ACos(in NDArray nd, Type dtype) => ACos(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise inverse cosine (arccos) using IL-generated kernels.
        /// </summary>
        public override NDArray ACos(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.ACos, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
