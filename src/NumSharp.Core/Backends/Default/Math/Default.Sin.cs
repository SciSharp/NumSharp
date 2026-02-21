using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Sin(in NDArray nd, Type dtype) => Sin(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise sine using IL-generated kernels.
        /// </summary>
        public override NDArray Sin(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Sin, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
