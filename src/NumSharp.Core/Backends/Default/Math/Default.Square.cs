using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Square(in NDArray nd, Type dtype) => Square(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise square (x*x) using IL-generated kernels.
        /// NumPy behavior: preserves input dtype (unlike trig functions which promote to float).
        /// </summary>
        public override NDArray Square(in NDArray nd, NPTypeCode? typeCode = null)
        {
            // np.square preserves input dtype (unlike sin/cos which promote to float)
            var outputType = typeCode ?? nd.GetTypeCode;
            return ExecuteUnaryOp(in nd, UnaryOp.Square, outputType);
        }
    }
}
