using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ACos(NDArray nd, Type dtype) => ACos(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise inverse cosine (arccos) using IL-generated kernels.
        /// </summary>
        public override NDArray ACos(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.ACos, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
