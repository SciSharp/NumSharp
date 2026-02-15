using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Floor(in NDArray nd, Type dtype) => Floor(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise floor using IL-generated kernels.
        /// </summary>
        public override NDArray Floor(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Floor, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
