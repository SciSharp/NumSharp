using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log10(NDArray nd, Type dtype) => Log10(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise log base 10 using IL-generated kernels.
        /// </summary>
        public override NDArray Log10(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Log10, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
