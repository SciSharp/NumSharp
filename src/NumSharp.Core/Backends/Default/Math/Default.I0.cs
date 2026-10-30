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
