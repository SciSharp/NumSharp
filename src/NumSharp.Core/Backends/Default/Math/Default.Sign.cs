using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Sign(NDArray nd, Type dtype) => Sign(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise sign function using IL-generated kernels.
        /// Returns -1, 0, or 1 based on input sign.
        /// NumPy behavior: preserves input dtype.
        /// </summary>
        public override NDArray Sign(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateWhereMask(where);

            // np.sign preserves input dtype (unlike trigonometric functions);
            // dtype= selects the loop instead and the input must reach it via
            // a same_kind cast (probed: sign(f8, dtype=i4) raises the input
            // cast error; sign(i4, dtype=f8) -> [0., 1., ...]).
            var loopType = typeCode ?? nd.GetTypeCode;

            // sign has NO bool loop (ufunc.types starts at 'b->b'; probed
            // verbatim on 2.4.2).
            if (loopType == NPTypeCode.Boolean)
                throw new TypeError(
                    "ufunc 'sign' did not contain a loop with signature matching types " +
                    "<class 'numpy.dtypes.BoolDType'> -> None");

            if (typeCode.HasValue)
                ValidateUnaryInputCast(nd.GetTypeCode, typeCode.Value, "sign");

            return ExecuteUnaryOp(nd, UnaryOp.Sign, loopType, @out, where);
        }
    }
}
