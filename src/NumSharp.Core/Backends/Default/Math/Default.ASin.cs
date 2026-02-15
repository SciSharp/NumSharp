using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ASin(in NDArray nd, Type dtype) => ASin(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise inverse sine (arcsin) using IL-generated kernels.
        /// </summary>
        public override NDArray ASin(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.ASin, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
