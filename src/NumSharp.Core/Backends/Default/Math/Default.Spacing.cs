using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Element-wise <c>np.spacing</c>: the distance from each value to the adjacent representable
        ///     value in the direction AWAY from zero (one ULP). IL-generated kernels — SIMD for
        ///     float32/float64 (a bit-increment + zero-mask), scalar for the promoted-int/Half/Decimal
        ///     paths. See <see cref="NumSharp.Utilities.NDSpacingMath"/> for the exact per-dtype semantics.
        /// </summary>
        /// <param name="nd">Input array; its dtype selects the float loop (see the promotion table below).</param>
        /// <param name="dtype">
        ///     Explicit loop dtype (NumPy ufunc <c>dtype=</c>): must be a FLOAT loop (Half/Single/Double,
        ///     or NumSharp's Decimal extension). An integer/bool or COMPLEX request has no loop — spacing
        ///     is float-only — and raises NumPy's "No loop matching the specified signature" error.
        /// </param>
        /// <param name="out">A location into which the result is stored (must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <returns>
        ///     Array of the same shape as <paramref name="nd"/>, dtype promoted per NumPy's width rule
        ///     (bool/int8/uint8→float16, int16/uint16/char→float32, int32+→float64; float/decimal preserved).
        /// </returns>
        /// <exception cref="IncorrectTypeException">
        ///     A complex <paramref name="dtype"/> request (no complex loop → "No loop matching …"), a complex
        ///     INPUT with no dtype override (→ "ufunc 'spacing' not supported for the input types …"), an
        ///     integer/bool <paramref name="dtype"/> (→ "No loop matching …"), or a complex input that cannot
        ///     same_kind-cast to a requested real loop (→ "Cannot cast ufunc 'spacing' input …").
        /// </exception>
        public override NDArray Spacing(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();

            // NumPy validation order: the where bool check is argument parsing — it precedes loop
            // resolution (matching arcsinh/ExecuteFloatTierBinary).
            ValidateWhereMask(where);

            // spacing has float-only loops (ee/ff/dd + NumSharp's decimal extension) — there is NO complex
            // loop, so a complex dtype= REQUEST is the "No loop matching" error (probed 2.4.2: same text as
            // an int/bool dtype=). This must run BEFORE ResolveUnaryFloatReturnType, which would otherwise
            // accept Complex (it is a valid loop for sin/exp/…, but not for spacing).
            if (typeCode == NPTypeCode.Complex)
                throw new IncorrectTypeException(
                    "No loop matching the specified signature and casting was found for ufunc spacing");

            // A complex INPUT with no dtype override cannot coerce to any float loop: NumPy raises the
            // TypeError "not supported for the input types" (probed 2.4.2). With a real dtype= given, the
            // cast is instead adjudicated by ResolveUnaryFloatReturnType → ValidateUnaryInputCast, which
            // yields NumPy's "Cannot cast ufunc 'spacing' input from complex128 to …" — so guard only the
            // no-dtype case here.
            if (nd.GetTypeCode == NPTypeCode.Complex && !typeCode.HasValue)
                throw new IncorrectTypeException(
                    "ufunc 'spacing' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            return ExecuteUnaryOp(nd, UnaryOp.Spacing, ResolveUnaryFloatReturnType(nd, typeCode, "spacing"), @out, where);
        }
    }
}
