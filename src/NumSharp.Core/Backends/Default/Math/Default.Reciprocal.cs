using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Reciprocal(NDArray nd, Type dtype) => Reciprocal(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise reciprocal (1/x) using IL-generated kernels.
        /// </summary>
        public override NDArray Reciprocal(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Reciprocal, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
