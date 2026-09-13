using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.frexp</c> — decompose each element into a normalized fraction and an integer power of two:
        ///     returns <c>(mantissa, exponent)</c> with <c>x == mantissa * 2^exponent</c> and the mantissa in
        ///     [0.5, 1) (0 / ±inf / NaN pass through as the mantissa). The mantissa carries the input's float
        ///     type; the exponent is ALWAYS int32 (NumPy's e-&gt;ei / f-&gt;fi / d-&gt;di loops).
        /// </summary>
        /// <param name="x">Input array. Integer/bool inputs promote to a float per NumPy's width rule
        /// (bool/int8/uint8-&gt;float16, int16/uint16/char-&gt;float32, int32+-&gt;float64); Half/Single/Double are
        /// preserved and Decimal is a NumSharp extension (computed through the double bridge). Complex is rejected.</param>
        /// <returns>
        ///     A tuple <c>(Mantissa, Exponent)</c>: the mantissa in the promoted float dtype and the int32
        ///     exponent, both C-contiguous with the shape of <paramref name="x"/>. (The public
        ///     <see cref="np.frexp"/> wrapper restores F-order when the input was F-contiguous.)
        /// </returns>
        /// <exception cref="IncorrectTypeException"><paramref name="x"/> is complex — frexp has no complex loop
        /// (NumPy raises the same "not supported for the input types" TypeError).</exception>
        /// <remarks>
        ///     Bit-identical to win-amd64 NumPy 2.4.2, including the platform C-runtime quirks its scalar
        ///     <c>npy_frexp</c> produces: <c>frexp(±inf)</c> and <c>frexp(NaN)</c> report exponent <c>-1</c>
        ///     (not 0), and a signalling NaN mantissa is quieted. See <see cref="DirectILKernelGenerator.FrexpHelper(double*,double*,int*,long)"/>.
        /// </remarks>
        public override (NDArray Mantissa, NDArray Exponent) Frexp(NDArray x, NDArray out1 = null, NDArray out2 = null, NDArray where = null)
        {
            // NumPy validation order: where parse -> input coercion / loop resolution -> out cast.
            ValidateWhereMask(where);
            // frexp is float-only (e/f/d/g loops). A complex input coerces to no loop — NumPy's TypeError.
            if (x.GetTypeCode == NPTypeCode.Complex)
                throw new IncorrectTypeException(
                    "ufunc 'frexp' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            // The mantissa's dtype is the input's float tier (int/bool promote; float/decimal preserved).
            var floatType = ResolveUnaryFloatReturnType(x, null, "frexp");
            // out1 (mantissa) takes a same_kind cast from the mantissa loop dtype; out2 (exponent) from int32
            // (so an int64 exponent-out is accepted, an int32 mantissa-out is not). Matches NumPy's ufunc rules.
            if (out1 is not null)
                ValidateOutCast(floatType, out1.typecode, "frexp");
            if (out2 is not null)
                ValidateOutCast(NPTypeCode.Int32, out2.typecode, "frexp");

            // Read the input as a C-contiguous buffer at the mantissa dtype (reads through ANY layout — a
            // strided/transposed/broadcast view is materialized in logical C-order; a no-op passthrough when x
            // already IS that contiguous dtype). The kernel is OUT OF PLACE (src -> mant, exp) so the two
            // outputs share one fresh C-contiguous layout: writing them in place off an F-contiguous Cast copy
            // would misalign the mantissa and exponent buffers (Cast preserves the input's contiguity, but a
            // fresh int32 exponent is always C-contiguous). np.frexp then relabels both to F together.
            var cShape = new Shape(x.shape);
            var src = MaterializeContiguous(x, cShape, floatType);
            var mant = new NDArray(floatType, cShape, fillZeros: false);
            var exp = new NDArray(NPTypeCode.Int32, cShape, fillZeros: false);

            long len = cShape.size;
            unsafe
            {
                switch (floatType)
                {
                    case NPTypeCode.Double:
                        DirectILKernelGenerator.FrexpHelper((double*)src.Address, (double*)mant.Address, (int*)exp.Address, len);
                        break;
                    case NPTypeCode.Single:
                        DirectILKernelGenerator.FrexpHelper((float*)src.Address, (float*)mant.Address, (int*)exp.Address, len);
                        break;
                    case NPTypeCode.Half:
                        DirectILKernelGenerator.FrexpHelper((Half*)src.Address, (Half*)mant.Address, (int*)exp.Address, len);
                        break;
                    case NPTypeCode.Decimal:
                        DirectILKernelGenerator.FrexpHelper((decimal*)src.Address, (decimal*)mant.Address, (int*)exp.Address, len);
                        break;
                    default:
                        // ResolveUnaryFloatReturnType only yields Half/Single/Double/Decimal for a non-complex
                        // input, so this is unreachable — kept as a guard against a future tier change.
                        throw new NotSupportedException($"frexp does not support dtype {floatType}");
                }
            }

            // Reclaim the materialized source (a fresh copy for non-contiguous / cast inputs); never the caller's x.
            if (!ReferenceEquals(src, x)) src.Dispose();

            // ufunc out=/where= tail: with no out, return the fresh (mant, exp); otherwise write each into its
            // caller-supplied out with a same_kind cast, masked by where= (masked-off slots keep prior contents),
            // and return the out arrays. The mantissa and exponent outs are independent (either may be supplied).
            if (out1 is null && out2 is null)
                return (mant, exp);
            NDArray rm = mant, re = exp;
            if (out1 is not null) { np.copyto(out1, mant, "same_kind", where); mant.Dispose(); rm = out1; }
            if (out2 is not null) { np.copyto(out2, exp, "same_kind", where); exp.Dispose(); re = out2; }
            return (rm, re);
        }
    }
}
