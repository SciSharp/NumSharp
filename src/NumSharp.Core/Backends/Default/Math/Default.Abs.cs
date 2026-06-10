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
        public override NDArray Abs(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            var inputType = nd.GetTypeCode;

            // NumPy: np.abs(complex) returns float64 (the magnitude), not complex
            // The IL kernel handles Complex→Double type change.
            // dtype= names the loop's OUTPUT for complex absolute (D->d): float
            // kinds select the magnitude dtype; integer/bool/complex requests
            // have no loop (probed 2.4.2: abs(c128, dtype=f64) → [5.],
            // dtype=c128 → "No loop matching ... ufunc absolute").
            if (inputType == NPTypeCode.Complex)
            {
                if (typeCode.HasValue && (typeCode.Value < NPTypeCode.Single || typeCode.Value == NPTypeCode.Complex))
                    throw new IncorrectTypeException(
                        "No loop matching the specified signature and casting was found for ufunc absolute");
                var outputType = typeCode ?? NPTypeCode.Double;
                return ExecuteUnaryOp(nd, UnaryOp.Abs, outputType, @out, where);
            }

            // dtype= runs the loop in that dtype: the input must reach it via
            // a same_kind cast (abs(f64, dtype=i32) raises, probed 2.4.2).
            if (typeCode.HasValue)
                ValidateUnaryInputCast(inputType, typeCode.Value, "absolute");

            // np.abs preserves input dtype (unlike trigonometric functions)
            // Only use explicit typeCode if provided, otherwise keep input type
            var resultType = typeCode ?? inputType;

            // Unsigned types are already non-negative - just return a copy with type cast
            // (with out=/where= the iterator route runs so the result lands in out).
            if (inputType.IsUnsigned() && @out is null && where is null)
            {
                return Cast(nd, resultType, copy: true);
            }

            return ExecuteUnaryOp(nd, UnaryOp.Abs, resultType, @out, where);
        }
    }
}
