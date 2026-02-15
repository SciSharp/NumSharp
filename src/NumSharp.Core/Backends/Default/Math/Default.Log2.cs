using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log2(in NDArray nd, Type dtype) => Log2(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise log base 2 using IL-generated kernels.
        /// </summary>
        public override NDArray Log2(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Log2, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
