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

            // np.abs(complex) returns the float64 magnitude. NumPy registers exactly the
            // D->d loop for complex 'absolute' (complex128 in, float64 magnitude out); the
            // IL kernel performs the Complex→Double reduction. NumSharp's only complex dtype
            // is complex128, so the loop output is always float64 and a dtype= request can
            // only ever name that same float64 — every other request fails (probed 2.4.2):
            //   • dtype=None / float64 → float64 magnitude (the D->d loop)
            //   • dtype=complex128     → no D->D loop → "No loop matching…" (TypeError)
            //   • any other dtype      → complex→X is never a same_kind cast →
            //                            "Cannot cast ufunc 'absolute' input from
            //                             dtype('complex128') to dtype('X')…" (UFuncTypeError)
            // Resolve by VALUE, not by enum order: the NPTypeCode ints predate NEP50 and
            // put Half=16/Decimal=15 ABOVE Single=13, so the old `typeCode < Single` guard
            // silently accepted Half/Decimal/Single and mislabeled the integer/bool requests
            // as "No loop" instead of NumPy's "Cannot cast" UFuncTypeError.
            if (inputType == NPTypeCode.Complex)
            {
                if (typeCode.HasValue && typeCode.Value != NPTypeCode.Double)
                {
                    if (typeCode.Value == NPTypeCode.Complex)
                        throw new IncorrectTypeException(
                            "No loop matching the specified signature and casting was found for ufunc absolute");

                    // complex→{real,int,bool} is never a same_kind cast, so this always
                    // raises NumPy's verbatim "Cannot cast ufunc 'absolute' input …" text.
                    ValidateUnaryInputCast(inputType, typeCode.Value, "absolute");
                }

                return ExecuteUnaryOp(nd, UnaryOp.Abs, NPTypeCode.Double, @out, where);
            }

            // absolute registers a loop for every real/int/bool dtype (D->D) plus the
            // complex magnitude loops (D->d) — but NO loop with a COMPLEX output. So a
            // real input with dtype=complex128 never resolves even though real→complex IS
            // a same_kind cast (which would wrongly satisfy the generic check below). NumPy
            // reports the signature-specific "did not contain a loop" form here, distinct
            // from the complex-INPUT "No loop matching the specified signature" above
            // (both probed 2.4.2).
            if (typeCode == NPTypeCode.Complex)
                throw new TypeError(
                    "ufunc 'absolute' did not contain a loop with signature matching types " +
                    $"<class 'numpy.dtypes.{NumPyDTypeClassName(inputType)}'> -> <class 'numpy.dtypes.Complex128DType'>");

            // dtype= runs the loop in that dtype: the input must reach it via
            // a same_kind cast (abs(f64, dtype=i32) raises, probed 2.4.2).
            if (typeCode.HasValue)
                ValidateUnaryInputCast(inputType, typeCode.Value, "absolute");

            // np.abs preserves input dtype (unlike trigonometric functions)
            // Only use explicit typeCode if provided, otherwise keep input type
            var resultType = typeCode ?? inputType;

            // Unsigned/bool/char abs is identity, so a plain dtype-preserving copy is exact —
            // but ONLY when no dtype change is requested. With a *narrowing* (or equal-width)
            // signed dtype= the cast happens BEFORE the loop: abs(uint8 200, dtype=int8)
            // casts 200→-56 then takes abs → 56, whereas a bare copy would keep -56 (probed
            // 2.4.2). Those cases — and any out=/where= — must run the cast-then-abs kernel
            // via ExecuteUnaryOp. (Widening signed/float dtype= is value-identical either
            // way, but routing it through the kernel keeps one code path.)
            if (inputType.IsUnsigned() && resultType == inputType && @out is null && where is null)
            {
                return Cast(nd, resultType, copy: true);
            }

            return ExecuteUnaryOp(nd, UnaryOp.Abs, resultType, @out, where);
        }
    }
}
