using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Element-wise population count of the magnitude (np.bitwise_count): the number of 1-bits in
        ///     <c>|x|</c> for each element, always producing a <b>uint8</b> result. INTEGER/BOOL/CHAR ONLY —
        ///     float/complex/decimal/half inputs have no loop and raise (NumPy 2.4.2 parity). Signed negatives
        ///     count the magnitude (<c>bitwise_count(-1) == 1</c>, not 8; <c>bitwise_count(int8 -128) == 1</c>).
        /// </summary>
        /// <param name="nd">Input array (bool/byte/sbyte/int16/uint16/int32/uint32/int64/uint64/char).</param>
        /// <param name="dtype">Output dtype request. Only uint8 (or null) is a valid loop; anything else raises the no-loop error.</param>
        /// <param name="@out">Optional output (NumPy ufunc out=); the uint8 result is cast same_kind into it (e.g. an int32 out is fine).</param>
        /// <param name="where">Optional bool write mask (NumPy ufunc where=).</param>
        /// <returns>A uint8 array of set-bit counts (or <paramref name="@out"/> when supplied).</returns>
        /// <exception cref="ArgumentException">
        ///     <paramref name="where"/> is a non-bool mask, or an <paramref name="@out"/>/dtype input cannot cast
        ///     same_kind (verbatim UFuncTypeError text).
        /// </exception>
        /// <exception cref="IncorrectTypeException"><paramref name="dtype"/> is given and is not uint8 (no matching loop).</exception>
        /// <exception cref="TypeError">The input dtype is not integer/bool/char and cannot be coerced to a supported loop.</exception>
        public override NDArray BitwiseCount(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order (probed 2.4.2): where parse -> loop resolution (dtype) -> input cast.
            ValidateWhereMask(where);

            // dtype= for bitwise_count selects the OUTPUT loop, and every loop outputs uint8 — so uint8 is
            // the only accepted request (independent of input width; the input keeps its own loop). Any
            // other dtype has no matching loop. Probed: bitwise_count(i4, dtype=i4) and (f8, dtype=i4) both
            // raise the no-loop TypeError, and it is reported BEFORE the input-cast check below.
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            if (typeCode.HasValue && typeCode.Value != NPTypeCode.Byte)
                throw new IncorrectTypeException(
                    "No loop matching the specified signature and casting was found for ufunc bitwise_count");

            var inputType = nd.typecode;
            bool inputIntLike = inputType == NPTypeCode.Boolean || NDExprTypeRules.IsIntegerKind(inputType);
            if (!inputIntLike)
            {
                // A non-int input with dtype=uint8 is an input CAST failure (float64 -> uint8 is not
                // same_kind), reported with the input-cast text; without dtype= it is the "not supported for
                // the input types" coercion error. Both probed verbatim on 2.4.2.
                if (typeCode.HasValue)
                    ValidateUnaryInputCast(inputType, NPTypeCode.Byte, "bitwise_count");
                throw new TypeError(
                    "ufunc 'bitwise_count' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");
            }

            // Output is always uint8; the op runs on the INPUT dtype (EmitsResultFromInputType), so no
            // input cast happens even for a wide integer — the popcount is computed at full width.
            return ExecuteUnaryOp(nd, UnaryOp.BitwiseCount, NPTypeCode.Byte, @out, where);
        }
    }
}
