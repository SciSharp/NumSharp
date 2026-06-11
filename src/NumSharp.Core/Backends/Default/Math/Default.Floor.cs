using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Floor(NDArray nd, Type dtype) => Floor(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise floor using IL-generated kernels.
        /// NumPy: for integer dtypes, floor is a no-op that preserves the input dtype.
        /// </summary>
        public override NDArray Floor(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateWhereMask(where);

            // NumPy registers IDENTITY loops for floor on every bool/
            // integer dtype ('?->?','b->b',...,'Q->Q'), so the loop dtype is
            // PRESERVED for int-like inputs. dtype= selects the loop and the
            // input must reach it via a same_kind cast (floor(f8, dtype=i4)
            // raises the input cast error; dtype=i8/f4 on i4 are fine) --
            // all probed on 2.4.2.
            var inputType = nd.GetTypeCode;
            if (typeCode.HasValue)
                ValidateUnaryInputCast(inputType, typeCode.Value, "floor");

            bool intLike = inputType == NPTypeCode.Boolean || inputType.IsInteger();
            if (!typeCode.HasValue && intLike && @out is null && where is null)
                return Cast(nd, inputType, copy: true); // fast identity memcpy

            var loopType = typeCode ?? (intLike ? inputType : ResolveUnaryReturnType(nd, null));
            return ExecuteUnaryOp(nd, UnaryOp.Floor, loopType, @out, where);
        }
    }
}
