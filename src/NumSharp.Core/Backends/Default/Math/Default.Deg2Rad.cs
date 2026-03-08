using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Deg2Rad(in NDArray nd, Type dtype) => Deg2Rad(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise degrees to radians conversion using IL-generated kernels.
        /// Computes x * (π/180).
        /// </summary>
        public override NDArray Deg2Rad(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Deg2Rad, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
