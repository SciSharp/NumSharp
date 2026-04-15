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
            if (inputType == NPTypeCode.Complex)
            {
                var outputType = typeCode ?? NPTypeCode.Double;
                return ExecuteComplexAbs(nd, outputType);
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

        /// <summary>
        /// Execute abs for complex arrays - returns float64 magnitude.
        /// </summary>
        private NDArray ExecuteComplexAbs(NDArray nd, NPTypeCode outputType)
        {
            var result = new NDArray(outputType, nd.Shape.Clean(), false);

            // Use iterator for complex abs since it changes type
            var inputIter = nd.AsIterator<System.Numerics.Complex>();
            var outputIter = result.AsIterator<double>();

            while (inputIter.HasNext())
            {
                var c = inputIter.MoveNext();
                outputIter.MoveNextReference() = System.Numerics.Complex.Abs(c);
            }

            // Cast to requested output type if not double
            if (outputType != NPTypeCode.Double)
            {
                return Cast(result, outputType, copy: false);
            }

            return result;
        }
    }
}
