using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Square(NDArray nd, Type dtype) => Square(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise square (x*x) using IL-generated kernels.
        /// NumPy behavior: preserves input dtype (unlike trig functions which promote to float).
        /// </summary>
        public override NDArray Square(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // dtype= runs the loop in that dtype: the input must reach it via
            // a same_kind cast (square(f64, dtype=i32) raises, probed 2.4.2).
            if (typeCode.HasValue)
                ValidateUnaryInputCast(nd.GetTypeCode, typeCode.Value, "square");

            // np.square preserves input dtype (unlike sin/cos which promote to float), EXCEPT
            // bool: square has no bool loop, so square(bool) -> int8 (probed 2.4.2).
            var outputType = typeCode ?? (nd.GetTypeCode == NPTypeCode.Boolean ? NPTypeCode.SByte : nd.GetTypeCode);
            return ExecuteUnaryOp(nd, UnaryOp.Square, outputType, @out, where);
        }
    }
}
