using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Abs(in NDArray nd, Type dtype) => Abs(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise absolute value using IL-generated kernels.
        /// </summary>
        public override NDArray Abs(in NDArray nd, NPTypeCode? typeCode = null)
        {
            // Unsigned types are already non-negative - just return a copy with type cast
            var outputType = ResolveUnaryReturnType(nd, typeCode);
            if (nd.typecode.IsUnsigned())
            {
                return Cast(nd, outputType, copy: true);
            }

            return ExecuteUnaryOp(in nd, UnaryOp.Abs, outputType);
        }
    }
}
