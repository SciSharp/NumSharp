using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise power at a MINIMUM precision of float64 (np.float_power).
        ///
        /// float_power is <see cref="Power"/> restricted to NumPy's two floating loops — float64
        /// (<c>dd->d</c>) and complex128 (<c>DD->D</c>) — so it is implemented AS Power on a forced
        /// float loop rather than as a new kernel: the arithmetic is bit-identical, so reusing the
        /// already-bit-exact Power path (npy_cpow for complex, the float64 pow loop for reals) is
        /// what guarantees parity. The two behaviours that make float_power DIFFERENT from Power are
        /// both properties of loop SELECTION, handled here before delegating:
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
        /// shared validators; the delegated <see cref="Power"/> re-runs the identical checks (naming
        /// 'power'), but since every one already passed it never raises, so the name never leaks.
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

            // Compute via the bit-exact Power engine on the resolved loop. When both operands ALREADY
            // carry the loop dtype (the common float64/complex128 case) delegate WITHOUT a dtype=
            // override so Power's scalar-exponent fast paths engage — float_power(x, 2.0) becomes x*x
            // and float_power(x, 0.5) becomes sqrt(x) instead of a per-element Math.Pow, and the
            // natural result type already IS the loop, so the value is identical. Otherwise force the
            // loop so int/float16/float32/bool/char/decimal and mixed operands compute (and, for the
            // narrow-natural cases, promote) at float64/complex128 exactly as NumPy's float_power does.
            bool operandsAlreadyLoop = lhs.GetTypeCode == loop && rhs.GetTypeCode == loop;
            return operandsAlreadyLoop
                ? Power(lhs, rhs, null, @out, where)
                : Power(lhs, rhs, loop, @out, where);
        }
    }
}
