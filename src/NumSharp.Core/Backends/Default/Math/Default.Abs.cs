using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Abs(NDArray nd, Type dtype) => Abs(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise absolute value using IL-generated kernels.
        /// NumPy behavior: preserves input dtype (unlike sin/cos which promote to float).
        /// </summary>
        public override NDArray Abs(NDArray nd, NPTypeCode? typeCode = null)
        {
            // np.abs preserves input dtype (unlike trigonometric functions)
            // Only use explicit typeCode if provided, otherwise keep input type
            var outputType = typeCode ?? nd.GetTypeCode;

            // Unsigned types are already non-negative - just return a copy with type cast
            if (nd.typecode.IsUnsigned())
            {
                return Cast(nd, outputType, copy: true);
            }

            return ExecuteUnaryOp(nd, UnaryOp.Abs, outputType);
        }
    }
}
