using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.ldexp</c> — compose <c>x1 * 2^x2</c> element-wise (the inverse of <see cref="Frexp"/>).
        ///     <paramref name="x1"/> is the mantissa/base (float family; integer/bool promote per NumPy's
        ///     width rule) and <paramref name="x2"/> is an INTEGER exponent — the two do NOT promote to a
        ///     common dtype (NumPy's ei-&gt;e / fi-&gt;f / di-&gt;d loops keep the exponent an int), so the
        ///     result dtype is purely <paramref name="x1"/>'s float tier.
        /// </summary>
        /// <param name="x1">Mantissa/base values. bool/int8/uint8-&gt;float16, int16/uint16/char-&gt;float32,
        /// int32+-&gt;float64; Half/Single/Double preserved; Decimal is a NumSharp extension (double bridge).
        /// Complex is rejected.</param>
        /// <param name="x2">Integer exponents (bool, int8/16/32/64, uint8/16/32, or char). An exponent outside
        /// the C-int range is clamped, so a huge magnitude overflows to ±inf / underflows to ±0 exactly as
        /// NumPy's int64 loop does. uint64, float and complex exponents are rejected.</param>
        /// <returns><c>x1 * 2^x2</c> broadcast to the common shape, in <paramref name="x1"/>'s float tier.
        /// F-contiguous inputs yield an F-contiguous result.</returns>
        /// <exception cref="IncorrectTypeException"><paramref name="x1"/> is complex, or <paramref name="x2"/> is
        /// not a supported integer exponent (uint64/float/complex) — NumPy raises the same "not supported for
        /// the input types" TypeError.</exception>
        /// <remarks>Bit-identical to win-amd64 NumPy 2.4.2 (<see cref="Math.ScaleB(double,int)"/> IS <c>ldexp</c>).</remarks>
        public override NDArray Ldexp(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order: where parse -> input coercion / loop resolution -> out cast -> shape.
            ValidateWhereMask(where);
            // ldexp has float×int loops only; a complex mantissa or a non-integer exponent coerces to no loop.
            if (x1.GetTypeCode == NPTypeCode.Complex)
                throw LdexpNoLoop();
            ValidateLdexpExponent(x2);

            var floatType = ResolveUnaryFloatReturnType(x1, null, "ldexp"); // x2 has no say in the result dtype
            // out= must accept a same_kind cast from the loop dtype (a float32 result up-casts into a float64 out).
            if (@out is not null)
                ValidateOutCast(floatType, @out.typecode, "ldexp");

            // Broadcast to the common result shape (the exponent never widens the mantissa's dtype).
            var (xShape, _) = Broadcast(x1.Shape, x2.Shape);
            var cleanShape = xShape.Clean();
            var result = new NDArray(floatType, cleanShape, fillZeros: false);

            long n = cleanShape.size;
            if (n == 0)
                return FinishLdexpOut(result, @out, where); // empty broadcast -> empty result

            // Materialize x1 broadcast to the result shape as a contiguous buffer of the mantissa dtype
            // (reads through any layout; a no-op passthrough when x1 is already that contiguous dtype).
            var xC = MaterializeContiguous(x1, cleanShape, floatType);

            unsafe
            {
                // Scalar-exponent fast path: a single (0-d or size-1) exponent scales the whole array without
                // materializing a per-element exponent buffer — the common np.ldexp(array, k) shape. Also
                // covers scalar×scalar (n == 1). Convert.ToInt64 reads the one value at any integer dtype.
                if (x2.size == 1)
                {
                    int e = DirectILKernelGenerator.ClampExp(Convert.ToInt64(x2.GetAtIndex(0)));
                    LdexpDispatchScalar(floatType, xC, e, result, n);
                }
                else if (ExponentFitsInt32(x2.GetTypeCode))
                {
                    // Common case: an int32-or-narrower exponent always fits the C-int range, so materialize
                    // it to contiguous int32 (a no-op passthrough for an int32 source) and skip the clamp —
                    // half the exponent traffic of the int64 path (the perf-critical branch at scale).
                    var expC = MaterializeContiguous(x2, cleanShape, NPTypeCode.Int32);
                    LdexpDispatchArray32(floatType, xC, expC, result, n);
                    if (!ReferenceEquals(expC, x2)) expC.Dispose();
                }
                else
                {
                    // uint32/int64 exponent: materialize to contiguous int64 (widens losslessly) and let the
                    // kernel clamp to int range per element (a value beyond ±MAX_INT overflows/underflows).
                    var expC = MaterializeContiguous(x2, cleanShape, NPTypeCode.Int64);
                    LdexpDispatchArray64(floatType, xC, expC, result, n);
                    if (!ReferenceEquals(expC, x2)) expC.Dispose(); // reclaim the materialized exponent temp
                }
            }

            if (!ReferenceEquals(xC, x1)) xC.Dispose(); // reclaim the materialized mantissa temp (never the caller's x1)

            // NumPy layout preservation: an all-F operand set yields an F-contiguous result (mirrors DivMod/modf).
            // (Skipped when out= is supplied — the caller's out array owns the layout.)
            if (@out is null && AreAllOperandsStrictFContig(x1, x2, cleanShape))
            {
                var rf = result.copy('F');
                result.Dispose();
                return rf;
            }
            return FinishLdexpOut(result, @out, where);
        }

        /// <summary>
        /// Apply the ufunc <c>out=</c>/<c>where=</c> tail: with no <paramref name="out"/> the freshly computed
        /// <paramref name="result"/> is returned as-is; otherwise the result is written into <paramref name="out"/>
        /// with a same_kind cast, masked by <paramref name="where"/> (masked-off slots keep their prior contents),
        /// and <paramref name="out"/> is returned. Mirrors the copyto-based write-back the other ufuncs use.
        /// </summary>
        /// <param name="result">The freshly computed result (disposed when copied into <paramref name="out"/>).</param>
        /// <param name="out">The caller's output array, or null.</param>
        /// <param name="where">The boolean mask, or null (already validated).</param>
        /// <returns><paramref name="out"/> when supplied (with the masked result written in), else <paramref name="result"/>.</returns>
        private static NDArray FinishLdexpOut(NDArray result, NDArray @out, NDArray where)
        {
            if (@out is null)
                return result; // where= without out= leaves nothing to mask against — return the full result
            np.copyto(@out, result, "same_kind", where); // same_kind cast + masked write (masked-off keep prior)
            result.Dispose();
            return @out;
        }

        /// <summary>Dispatch the scalar-exponent ldexp kernel for the resolved mantissa dtype (see <see cref="Ldexp"/>).</summary>
        private static unsafe void LdexpDispatchScalar(NPTypeCode floatType, NDArray xC, int e, NDArray result, long n)
        {
            switch (floatType)
            {
                case NPTypeCode.Double: DirectILKernelGenerator.LdexpScalarExp((double*)xC.Address, e, (double*)result.Address, n); break;
                case NPTypeCode.Single: DirectILKernelGenerator.LdexpScalarExp((float*)xC.Address, e, (float*)result.Address, n); break;
                case NPTypeCode.Half: DirectILKernelGenerator.LdexpScalarExp((Half*)xC.Address, e, (Half*)result.Address, n); break;
                case NPTypeCode.Decimal: DirectILKernelGenerator.LdexpScalarExp((decimal*)xC.Address, e, (decimal*)result.Address, n); break;
                default: throw new NotSupportedException($"ldexp does not support dtype {floatType}");
            }
        }

        /// <summary>Dispatch the int32-exponent ldexp kernel (no clamp) for the resolved mantissa dtype.</summary>
        private static unsafe void LdexpDispatchArray32(NPTypeCode floatType, NDArray xC, NDArray expC, NDArray result, long n)
        {
            switch (floatType)
            {
                case NPTypeCode.Double: DirectILKernelGenerator.LdexpHelper((double*)xC.Address, (int*)expC.Address, (double*)result.Address, n); break;
                case NPTypeCode.Single: DirectILKernelGenerator.LdexpHelper((float*)xC.Address, (int*)expC.Address, (float*)result.Address, n); break;
                case NPTypeCode.Half: DirectILKernelGenerator.LdexpHelper((Half*)xC.Address, (int*)expC.Address, (Half*)result.Address, n); break;
                case NPTypeCode.Decimal: DirectILKernelGenerator.LdexpHelper((decimal*)xC.Address, (int*)expC.Address, (decimal*)result.Address, n); break;
                default: throw new NotSupportedException($"ldexp does not support dtype {floatType}");
            }
        }

        /// <summary>Dispatch the int64-exponent ldexp kernel (per-element clamp) for uint32/int64 exponents.</summary>
        private static unsafe void LdexpDispatchArray64(NPTypeCode floatType, NDArray xC, NDArray expC, NDArray result, long n)
        {
            switch (floatType)
            {
                case NPTypeCode.Double: DirectILKernelGenerator.LdexpHelper((double*)xC.Address, (long*)expC.Address, (double*)result.Address, n); break;
                case NPTypeCode.Single: DirectILKernelGenerator.LdexpHelper((float*)xC.Address, (long*)expC.Address, (float*)result.Address, n); break;
                case NPTypeCode.Half: DirectILKernelGenerator.LdexpHelper((Half*)xC.Address, (long*)expC.Address, (Half*)result.Address, n); break;
                case NPTypeCode.Decimal: DirectILKernelGenerator.LdexpHelper((decimal*)xC.Address, (long*)expC.Address, (decimal*)result.Address, n); break;
                default: throw new NotSupportedException($"ldexp does not support dtype {floatType}");
            }
        }

        /// <summary>True when an exponent of this dtype always fits int32 (bool + int8..int32 + char); a
        /// uint32/int64 exponent needs the int64+clamp path.</summary>
        /// <param name="t">The exponent operand's dtype.</param>
        /// <returns>Whether the int32 (no-clamp) kernel may be used.</returns>
        private static bool ExponentFitsInt32(NPTypeCode t) => t switch
        {
            NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16
                or NPTypeCode.UInt16 or NPTypeCode.Char or NPTypeCode.Int32 => true,
            _ => false, // UInt32, Int64 -> int64+clamp path
        };

        /// <summary>
        ///     Reject an unsupported <c>ldexp</c> exponent dtype (uint64, any float, complex) with NumPy's
        ///     verbatim TypeError. The accepted set is bool + the signed/unsigned integers that safely cast to
        ///     int64 (int8..int64, uint8..uint32) plus char (NumSharp's unsigned-16-bit extension).
        /// </summary>
        /// <param name="x2">The exponent operand to validate.</param>
        /// <exception cref="IncorrectTypeException">The exponent dtype names no ldexp loop.</exception>
        private static void ValidateLdexpExponent(NDArray x2)
        {
            switch (x2.GetTypeCode)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                    return;
                default: // UInt64 (can't safely cast to int64), Half/Single/Double/Decimal, Complex
                    throw LdexpNoLoop();
            }
        }

        /// <summary>NumPy's verbatim "not supported for the input types" TypeError for ldexp.</summary>
        private static IncorrectTypeException LdexpNoLoop() => new IncorrectTypeException(
            "ufunc 'ldexp' not supported for the input types, and the inputs " +
            "could not be safely coerced to any supported types according to the casting rule ''safe''");
    }
}
