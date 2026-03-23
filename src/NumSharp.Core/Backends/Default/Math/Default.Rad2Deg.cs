using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Rad2Deg(NDArray nd, Type dtype) => Rad2Deg(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise radians to degrees conversion using IL-generated kernels.
        /// Computes x * (180/π).
        /// </summary>
        public override NDArray Rad2Deg(NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Rad2Deg, ResolveUnaryReturnType(nd, typeCode));
        }
    }
}
