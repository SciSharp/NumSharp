using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise floor using IL-generated kernels.
        /// NumPy: for integer dtypes, floor is a no-op that preserves the input dtype.
        /// </summary>
        public override NDArray Floor(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            ValidateWhereMask(where);

            // NumPy registers IDENTITY loops for floor on every bool/
            // integer dtype ('?->?','b->b',...,'Q->Q'), so the loop dtype is
            // PRESERVED for int-like inputs. dtype= selects the loop and the
            // input must reach it via a same_kind cast (floor(f8, dtype=i4)
            // raises the input cast error; dtype=i8/f4 on i4 are fine) --
            // all probed on 2.4.2.
            var inputType = nd.GetTypeCode;

            // floor has no complex loop (float + identity-int loops only). A complex input with no
            // explicit dtype reaches no loop, so NumPy raises the generic ufunc TypeError (NOT a
            // NotSupportedException from the kernel) and validates the LOOP, not the data — so a
            // zero-size complex operand is rejected too (probed 2.4.2). Same guard shape as np.fabs;
            // resolves the oracle K4 (wording/type) and K5 (zero-size skip) excuses at once.
            if (!typeCode.HasValue && inputType == NPTypeCode.Complex)
                throw new TypeError(
                    "ufunc 'floor' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

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
