// NumPy's np.sinc (numpy/lib/_function_base_impl.py) as DefaultEngine's member: NumPy's four statements fused into
// one np.evaluate pass. The file sits beside its facade (Math/np.sinc.cs) while it calls np.*: ratchet R2 counts np.*
// calls against every file under Backends/ (the np-layer dtype plan, C3); it moves under Backends/Default/ in P6.

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

            // Build the fused tree in NumPy's exact order. `(NDExpr)x` wraps the array
            // as a leaf (no copy); `* Math.PI` and the eps/0.0 scalars are ConstNodes
            // (raw doubles downcast to the output precision on emit — the NEP 50
            // weak-scalar adoption, reproduced bit-for-bit). `t` (= pi*x) is built ONCE
            // and referenced by both the zero test and the kept value, and the SAME `y`
            // node feeds both sin(y) and the divisor — one shared subtree each, so
            // `sin(y)/y` is computed against one deterministic value.
            NDExpr t = (NDExpr)x * Math.PI;
            NDExpr y = NDExpr.Where(NDExpr.NotEqual(t, 0.0), t, eps);

            // One np.evaluate pass: the operand is read once, the result written once.
            return np.evaluate(NDExpr.Sin(y) / y);
        }
    }
}
