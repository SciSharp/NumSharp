// NumPy's np.sinc (numpy/lib/_function_base_impl.py) as DefaultEngine's member: NumPy's four statements fused into
// one pass of this engine's Evaluate (np.evaluate's member). The facade is Math/np.sinc.cs.

using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // Machine epsilon per output dtype — NumPy's `finfo(x.dtype).eps` for the
        // three float kinds, and `1e-20` for everything NumPy classes as non-'f'
        // (complex here; Decimal is NumSharp's own extension of that branch). These
        // are the exact IEEE values 2^-10 / 2^-23 / 2^-52, representable in each type.
        private const double SincEpsFloat16 = 0.0009765625;                 // 2^-10
        private const double SincEpsFloat32 = 1.1920928955078125e-07;       // 2^-23
        private const double SincEpsFloat64 = 2.220446049250313e-16;        // 2^-52
        private const double SincEpsNonFloat = 1e-20;                       // NumPy's kind != 'f' branch

        /// <inheritdoc />
        public override NDArray Sinc(NDArray x)
        {
            // NumPy's sinc starts with pi * x, a Python float times text: multiply has no such loop. A StringDType x puts a
            // new-style DType into the key, so the dispatcher names the classes (pi is the weak _PyFloatDType); a fixed-width
            // x meets the legacy resolver, which names float64.
            if (Backends.Strings.NDStringUfuncs.IsVString(x))
                throw Backends.Strings.NDStringUfuncs.VStringNoLoop("multiply", new[] { DTypeRegistry.PyFloat.ToString(), x.dtype.Meta.ToString() });
            if (Backends.Strings.NDStringUfuncs.IsString(x)) throw Backends.Strings.NDStringUfuncs.NoLoop("multiply", np.float64, x.dtype);
            // NumPy's implicit `asanyarray(None)` failure, given a clear identity here.
            if (x is null)
                throw new ArgumentNullException(nameof(x));

            // eps depends on the OUTPUT dtype = dtype(pi*x). Integers/bool/char all
            // promote to float64 (weak-float pi), so they take the float64 branch via
            // the switch default; complex/decimal take NumPy's non-'f' 1e-20 branch.
            double eps = x.typecode switch
            {
                NPTypeCode.Half => SincEpsFloat16,
                NPTypeCode.Single => SincEpsFloat32,
                NPTypeCode.Complex => SincEpsNonFloat,
                NPTypeCode.Decimal => SincEpsNonFloat,
                _ => SincEpsFloat64,
            };

            // NumPy's sinc is four ufunc calls (pi * x, np.where, sin, /), so its result has the ufunc layout of x: plain C
            // or F for a contiguous x (the last calls take the trivial loop), the iterator's K order otherwise (a permuted x
            // stays permuted, a broadcast one is C). The fused pass writes C order, so a non-C layout runs it over x viewed
            // in that layout's axis order and relabels the result (no copy).
            if (TryUnaryNonCLayout(x, inputCast: false, out var layout))
                return ComputeInLayout(x, layout, (Engine: this, Eps: eps), static (xv, s) => s.Engine.SincFused(xv, s.Eps));
            return SincFused(x, eps);
        }

        /// <summary>
        ///     The fused sinc pass: <c>sin(y) / y</c> with <c>y = where(pi * x != 0, pi * x, eps)</c>, one np.evaluate
        ///     program, read once and written once in C order.
        /// </summary>
        /// <param name="x">The operand (a numeric array; text was refused by the caller).</param>
        /// <param name="eps">NumPy's substitute for a zero <c>pi * x</c>: <c>finfo(dtype(pi * x)).eps</c> for a float
        ///     result, 1e-20 otherwise.</param>
        /// <returns>A fresh array of <c>dtype(pi * x)</c>, C-ordered (np.evaluate's layout for a C-ordered or strided
        ///     operand).</returns>
        private NDArray SincFused(NDArray x, double eps)
        {
            // Build the fused tree in NumPy's exact order. `(NDExpr)x` wraps the array
            // as a leaf (no copy); `* Math.PI` and the eps/0.0 scalars are ConstNodes
            // (raw doubles downcast to the output precision on emit — the NEP 50
            // weak-scalar adoption, reproduced bit-for-bit). `t` (= pi*x) is built ONCE
            // and referenced by both the zero test and the kept value, and the SAME `y`
            // node feeds both sin(y) and the divisor — one shared subtree each, so
            // `sin(y)/y` is computed against one deterministic value.
            NDExpr t = (NDExpr)x * Math.PI;
            NDExpr y = NDExpr.Where(NDExpr.NotEqual(t, 0.0), t, eps);

            // One fused pass (np.evaluate's member, its default options): the operand is read once, the result written
            // once.
            return Evaluate(NDExpr.Sin(y) / y);
        }
    }
}
