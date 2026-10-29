using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise power at a MINIMUM precision of float64 (np.float_power).
        ///
        /// float_power is the power KERNEL restricted to NumPy's two floating loops — float64
        /// (<c>dd->d</c>) and complex128 (<c>DD->D</c>) — so it runs that kernel on a forced float
        /// loop rather than a new one: the arithmetic is bit-identical (npy_cpow for complex, the
        /// CRT pow for reals). It does NOT go through <see cref="Power"/>, whose float loops take
        /// NumPy's scalar-exponent fast path (x*x, sqrt, 1/x for a zero exponent step): NumPy's
        /// float_power loop is a plain pow loop with no such branch. The two behaviours that make
        /// float_power DIFFERENT from Power's integer loops are both properties of loop SELECTION,
        /// handled here before computing:
        ///   1. Every real input (bool/int/float16/float32/decimal/char) promotes to float64 and a
        ///      complex operand to complex128 — never an integer loop. That is why float_power(2, -1)
        ///      is 0.5 and does NOT raise Power's "Integers to negative integer powers are not
        ///      allowed": the loop is float64, and the negative-exponent guard fires only on an
        ///      INTEGER loop (see <see cref="Power"/>).
        ///   2. dtype= may select ONLY float64 or complex128; any other request has no loop.
        ///
        /// Error taxonomy and ORDER are NumPy's (probed 2.4.2): read-only out → non-bool where →
        /// dtype-has-no-loop → input cannot cast to the loop (complex ->float64) → out cannot cast
        /// from the loop → shape. Each error is raised HERE with the 'float_power' ufunc name via the
        /// shared validators; the kernel route (<see cref="ExecuteBinaryOp"/>) re-runs the identical
        /// checks (naming 'power'), but since every one already passed it never raises, so the name never
        /// leaks — except the broadcast check, whose text names no ufunc.
        /// </summary>
        /// <param name="lhs">The bases (any dtype; cast to the resolved float loop).</param>
        /// <param name="rhs">The exponents (any dtype; cast to the resolved float loop).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>); only float64 or complex128 are valid.</param>
        /// <param name="out">Destination (NumPy ufunc <c>out=</c>); must be same_kind-castable from the loop dtype.</param>
        /// <param name="where">Boolean write-mask (NumPy ufunc <c>where=</c>).</param>
        /// <returns>Bases raised to exponents at float64/complex128 precision, or <paramref name="out"/> when supplied.</returns>
        /// <exception cref="IncorrectTypeException"><paramref name="dtype"/> is neither float64 nor complex128 (no matching loop).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is non-bool; a complex input cannot
        ///     same_kind-cast to a requested float64 loop; <paramref name="out"/> is read-only or not same_kind-castable
        ///     from the loop dtype; or the operand shapes do not broadcast together.</exception>
        public override NDArray FloatPower(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // (1) A read-only out is rejected first — NumPy reports "output array is read-only" ahead
            // of every where/dtype/cast/shape error (probed 2.4.2), and the kernel writes @out
            // directly, so this must precede any path that could compute into it.
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");

            // (2) where must be exactly bool (NumPy casts the mask with the 'safe' rule).
            ValidateWhereMask(where);

            // (3/4) Resolve the float loop. float_power has ONLY float64 and complex128 loops, so a
            // complex operand forces complex128 and everything else forces float64. An explicit
            // dtype= must name one of those two loops (else "No loop matching…", a TypeError) and the
            // inputs must same_kind-cast INTO it — the one reachable input-cast failure is a complex
            // base/exponent against a requested float64 loop (complex ->float64 is not same_kind),
            // which NumPy reports as 'float_power' input 0/1. A real input always same_kind-casts to
            // either loop, so the auto (no-dtype) branch cannot raise here.
            NPTypeCode loop = (lhs.GetTypeCode == NPTypeCode.Complex || rhs.GetTypeCode == NPTypeCode.Complex)
                ? NPTypeCode.Complex
                : NPTypeCode.Double;
            if (dtype is not null)
            {
                NPTypeCode requested = dtype.GetTypeCode();
                if (requested != NPTypeCode.Double && requested != NPTypeCode.Complex)
                    throw new IncorrectTypeException(
                        "No loop matching the specified signature and casting was found for ufunc float_power");
                ValidateBinaryInputCasts(lhs.GetTypeCode, rhs.GetTypeCode, requested, "float_power");
                loop = requested;
            }

            // (5) out must be reachable from the loop result dtype by a same_kind cast.
            if (@out is not null)
                ValidateOutCast(loop, @out.typecode, "float_power");

            // Compute with the power KERNEL on the resolved loop — pow (npy_cpow for complex) on every element, never
            // np.power's scalar-exponent fast path: NumPy's float_power loop is PyUFunc_dd_d / PyUFunc_DD_D over pow
            // (generate_umath.py: TD('dgDG', f='pow')), which has no stride-0 branch, so float_power(x, 2.0) is
            // pow(x, 2.0) (757.7617475757002 ** 2 = ...03c2, where np.power's x*x is ...03c1), float_power(-0.0, 0.5)
            // is +0.0 and float_power(sNaN, 1.0) is a quiet NaN (probed 2.4.2). Every check above already passed, and
            // the kernel route repeats them under the 'power' name without raising. Forcing the loop makes
            // int/float16/float32/bool/char/decimal and mixed operands compute at float64/complex128, as NumPy's does.
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Power, @out, where, loop);
        }
    }
}
