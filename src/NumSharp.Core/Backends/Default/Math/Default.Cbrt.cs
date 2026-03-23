using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cbrt(NDArray nd, Type dtype) => Cbrt(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise cube root using IL-generated kernels.
        /// Computes the cube root of each element.
        /// </summary>
        public override NDArray Cbrt(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Cbrt, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
