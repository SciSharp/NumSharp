// NumPy's np.i0 (numpy/lib/_function_base_impl.py) as DefaultEngine's member: the null check, the text parse and
// the dispatch by float width to the per-precision cephes kernels np.i0.cs keeps (np.kaiser shares the double one),
// each applied in one fused pass of this engine's Evaluate. The kernels stay on np as delegate fields the engine reads:
// NDExpr.Call keys its compiled kernel by the delegate's identity, and np.kaiser's window shares the double one.

using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray I0(NDArray x)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            // NumPy: `if x.dtype.kind != 'f': x = x.astype(float)` — a text array is parsed into float64 (each element by
            // float()), then I0 runs on the numbers.
            using var parsed = Backends.Strings.NDStringUfuncs.IsText(x) ? x.astype(np.float64, true, 'K', "unsafe") : null;
            if (parsed is not null)
                x = parsed;

            // NumPy's i0 ends in piecewise(abs(x), ...), whose result is zeros_like(abs(x)) filled in place: the K-order
            // twin of a ufunc result of x, so it has x's ufunc layout (plain C or F for a contiguous x, the iterator's
            // order for a permuted one; an integer x is first a K-order float copy, which keeps that order). The fused
            // pass writes C order, so a non-C layout runs it over x viewed in that layout's axis order and relabels.
            if (x.typecode != NPTypeCode.Complex && TryUnaryNonCLayout(x, inputCast: false, out var layout))
                return ComputeInLayout(x, layout, this, static (xv, engine) => engine.I0Fused(xv));
            return I0Fused(x);
        }

        /// <summary>
        ///     The fused I0 pass: the cephes kernel of the operand's precision domain, applied by one np.evaluate program.
        /// </summary>
        /// <param name="x">The operand (numbers; text was parsed by the caller).</param>
        /// <returns>A fresh array of the kernel's float dtype, C-ordered (np.evaluate's layout for a C-ordered or strided
        ///     operand).</returns>
        /// <exception cref="IncorrectTypeException">A complex128 operand (NumPy's "i0 not supported for complex values").</exception>
        private NDArray I0Fused(NDArray x)
        {
            // Dispatch by PRECISION DOMAIN, not per-dtype: the cephes recurrence is bit-exact only
            // when run at the input's own float width (see the header). Each arm is a genuinely
            // different arithmetic — mirroring the kernel's own EmitUnaryHalfOperation / decimal /
            // complex split — not a mechanical per-NPTypeCode copy. bool/int/Char fall to the double
            // arm, where NDExpr.Call converts the integer operand to double at the call edge (NumPy's
            // astype(float) before the computation), so no explicit cast is needed.
            switch (x.typecode)
            {
                case NPTypeCode.Complex:
                    throw new IncorrectTypeException("i0 not supported for complex values");
                case NPTypeCode.Half:
                    return Evaluate(NDExpr.Call(np._i0Half, (NDExpr)x));
                case NPTypeCode.Single:
                    return Evaluate(NDExpr.Call(np._i0Single, (NDExpr)x));
                case NPTypeCode.Decimal:
                    return Evaluate(NDExpr.Call(np._i0Decimal, (NDExpr)x));
                default:
                    return Evaluate(NDExpr.Call(np._i0Double, (NDExpr)x));
            }
        }
    }
}
