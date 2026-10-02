using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Element-wise floating-point absolute value (np.fabs) — the FLOAT-ONLY sibling of
        ///     <see cref="Abs(NDArray, DType, NDArray, NDArray)"/>. Unlike <c>absolute</c> it PROMOTES
        ///     integer/bool input to float (NEP50 width tier) and has NO complex loop, so a complex input
        ///     (or a <c>dtype=complex</c> request) is refused rather than reduced to a magnitude.
        /// </summary>
        /// <param name="nd">Input array (any dtype except Complex).</param>
        /// <param name="dtype">
        ///     Explicit loop dtype (NumPy ufunc <c>dtype=</c>). fabs has float loops ONLY (efdg), so a
        ///     bool/integer/char request — <b>or</b> a Complex request — raises the no-loop
        ///     <see cref="IncorrectTypeException"/>; a float request runs the loop at that precision and
        ///     the input must reach it by a <c>same_kind</c> cast (a complex input then raises "Cannot
        ///     cast ufunc 'fabs' input …").
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored (NumPy ufunc <c>out=</c>): joins the broadcast
        ///     without being stretched, must be <c>same_kind</c>-castable from the resolved float loop
        ///     dtype, and is returned as-is. If <c>null</c> (and <paramref name="where"/> null) a freshly
        ///     allocated float array is returned.
        /// </param>
        /// <param name="where">
        ///     Boolean mask (NumPy ufunc <c>where=</c>): only mask-true elements are computed/written;
        ///     masked-off <paramref name="out"/> slots keep prior contents. Must be bool.
        /// </param>
        /// <returns>
        ///     A float array (dtype per the NEP50 tier: bool/int8/uint8→float16, int16/uint16/char→float32,
        ///     int32/uint32/int64/uint64→float64, float16/float32/float64/decimal preserved), or
        ///     <paramref name="out"/> — each element the magnitude of the input (the sign bit cleared).
        /// </returns>
        /// <exception cref="ArgumentException">
        ///     <paramref name="where"/> is a non-bool array, or a float <paramref name="dtype"/> is requested
        ///     for a complex input (verbatim NumPy cast texts).
        /// </exception>
        /// <exception cref="IncorrectTypeException">
        ///     <paramref name="dtype"/> is a bool/integer/char or Complex dtype (no matching float loop).
        /// </exception>
        /// <exception cref="TypeError">
        ///     The input is Complex and no <paramref name="dtype"/> was given (fabs has no complex loop —
        ///     NumPy's "not supported for the input types" TypeError, same class as signbit/cbrt).
        /// </exception>
        /// <remarks>
        ///     Port of NumPy 2.4.2's <c>fabs</c> ufunc (<c>generate_umath.py</c>:
        ///     <c>TD(flts, f='fabs', astype={'e':'f'})</c> — float loops only, float16 via float32). fabs
        ///     IS <c>absolute</c> on the float loops — the C <c>fabs</c> clears the IEEE sign bit, so the
        ///     result is BIT-EXACT including NaN payloads (a negative NaN's sign is cleared, payload kept)
        ///     and <c>-0.0 → +0.0</c>. It therefore rides <see cref="UnaryOp.Fabs"/>, which aliases the
        ///     fully-SIMD <see cref="UnaryOp.Abs"/> kernel at every emit site (a distinct op only so the
        ///     ufunc name is "fabs"); the float output type resolved here (not <c>Abs</c>'s dtype-preserving
        ///     one) is what makes it fabs. The promoting path casts int→float BEFORE the abs (so <c>fabs(int.MinValue)</c> is the
        ///     exact float magnitude, never the wrapped integer abs). Decimal (no NumPy analog) is preserved
        ///     and computed via <see cref="Math.Abs(decimal)"/>, consistent with the other float-tier unaries.
        /// </remarks>
        public override NDArray Fabs(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            var inputType = nd.GetTypeCode;

            // NumPy validation order (probed 2.4.2): the where-bool check is argument parsing and
            // precedes loop resolution — fabs(complex, where=int) reports the where cast error, not
            // the complex-input error.
            ValidateWhereMask(where);

            // fabs has float loops ONLY — no complex loop, unlike absolute. A dtype=complex request
            // therefore never resolves regardless of the input (NumPy's signature "No loop matching…"
            // TypeError, probed 2.4.2). This must precede the shared ResolveUnaryFloatReturnType call:
            // that resolver serves sqrt (which DOES have a complex loop), and it would accept Complex
            // by ENUM VALUE (128 ≥ Single=13) rather than reject it.
            if (typeCode == NPTypeCode.Complex)
                throw new IncorrectTypeException(
                    "No loop matching the specified signature and casting was found for ufunc fabs");

            // A complex INPUT with no explicit dtype reaches no fabs loop: NumPy raises the generic
            // "not supported for the input types" TypeError (the same class as signbit). With a float
            // dtype= given the error instead comes from the same_kind input-cast check inside
            // ResolveUnaryFloatReturnType ("Cannot cast ufunc 'fabs' input from complex128 to <float>…"),
            // so only the dtype=None case is guarded here (both texts probed 2.4.2).
            if (typeCode is null && inputType == NPTypeCode.Complex)
                throw new TypeError(
                    "ufunc 'fabs' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            // Float-tier NEP50 promotion (bool/i8/u8→f16, i16/u16/char→f32, i32+→f64, floats/decimal
            // preserved); a bool/integer dtype= → "No loop…", a float dtype= with complex input →
            // "Cannot cast…". The resolved float output type is passed EXPLICITLY so ExecuteUnaryOp does
            // NOT take Abs's dtype-preserving branch — the abs kernel then casts int→float first and
            // clears the sign bit, i.e. fabs.
            var outputType = ResolveUnaryFloatReturnType(nd, typeCode, "fabs");
            // UnaryOp.Fabs shares Abs's kernel at every emit site (clear the IEEE sign bit) but is a
            // distinct op so UfuncName reports "fabs" (not "absolute") in the out= cast error. The
            // resolved float output type — not Abs's dtype-preserving one — is what makes it fabs.
            return ExecuteUnaryOp(nd, UnaryOp.Fabs, outputType, @out, where);
        }
    }
}
