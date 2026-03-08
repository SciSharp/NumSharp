using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Square(in NDArray nd, Type dtype) => Square(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise square (x*x) using IL-generated kernels.
        /// </summary>
        public override NDArray Square(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Square, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
