using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Sign(in NDArray nd, Type dtype) => Sign(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise sign function using IL-generated kernels.
        /// Returns -1, 0, or 1 based on input sign.
        /// NumPy behavior: preserves input dtype.
        /// </summary>
        public override NDArray Sign(in NDArray nd, NPTypeCode? typeCode = null)
        {
            // np.sign preserves input dtype (unlike trigonometric functions)
            var outputType = typeCode ?? nd.GetTypeCode;
            return ExecuteUnaryOp(in nd, UnaryOp.Sign, outputType);
        }
    }
}
