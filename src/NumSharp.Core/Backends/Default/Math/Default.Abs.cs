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
        /// Exception: np.abs(complex) returns float64 (the magnitude).
        /// </summary>
        public override NDArray Abs(NDArray nd, NPTypeCode? typeCode = null)
        {
            var inputType = nd.GetTypeCode;

            // NumPy: np.abs(complex) returns float64 (the magnitude), not complex
            // The IL kernel handles Complex→Double type change
            if (inputType == NPTypeCode.Complex)
            {
                var outputType = typeCode ?? NPTypeCode.Double;
                return ExecuteUnaryOp(nd, UnaryOp.Abs, outputType);
            }

            // np.abs preserves input dtype (unlike trigonometric functions)
            // Only use explicit typeCode if provided, otherwise keep input type
            var resultType = typeCode ?? inputType;

            // Unsigned types are already non-negative - just return a copy with type cast
            if (inputType.IsUnsigned())
            {
                return Cast(nd, resultType, copy: true);
            }

            return ExecuteUnaryOp(nd, UnaryOp.Abs, resultType);
        }
    }
}
