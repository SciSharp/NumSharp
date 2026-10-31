using System;
using System.Numerics;

namespace NumSharp.Backends
{
    /// <summary>
    ///     NumSharp's <see cref="LinalgEngine"/>, the default of <see cref="TensorEngine.Linalg"/>: NumPy's
    ///     <c>numpy/linalg/_linalg.py</c> statements in NumPy's order - the argument checks (<c>_assert_stacked_2d</c>,
    ///     <c>_assert_stacked_square</c>, <c>_assert_finite</c>), the dtype rule (<c>_commonType</c>), the text refusals of
    ///     the <c>_umath_linalg</c> gufuncs' input casts and each function's tail - over ten <c>protected virtual</c>
    ///     kernels that stand for those gufuncs (<see cref="CholeskyKernel"/>, <see cref="DetKernel"/>,
    ///     <see cref="SlogdetKernel"/>, <see cref="EigKernel"/>, <see cref="EighKernel"/>, <see cref="InvKernel"/>,
    ///     <see cref="SolveKernel"/>, <see cref="LstsqKernel"/>, <see cref="QrKernel"/>, <see cref="SvdKernel"/>).
    /// </summary>
    /// <remarks>
    ///     <para><b>The kernels.</b> Each answers an empty operand (an empty matrix or a zero-size batch) itself, without
    ///     LAPACK (<see cref="LinAlgEmpty"/>), then asks the operand engine's <see cref="TensorEngine.Blas"/>, then falls
    ///     back: the LU four (det, slogdet, inv, solve) to the managed LU (<see cref="ManagedLu"/>, allclose to NumPy), the
    ///     other six to <see cref="OpenBlasMissingBackendException"/>. So the bundled OpenBLAS of
    ///     <c>NumSharp.Interop.OpenBLAS</c> reaches them through <see cref="TensorEngine.Blas"/>, and no subclass is needed
    ///     for it.</para>
    ///     <para><b>Deriving.</b> Override a kernel to compute a factorisation another way (MKL, a GPU, a solver of
    ///     your own): the public members keep running NumPy's statements around it, so the override receives validated
    ///     operands already cast to the loop dtype (float32, float64 or complex128) and its result goes through NumPy's
    ///     collapse, casts and result layouts. Call the base kernel for the operands it does not serve (an empty one,
    ///     another dtype). Assign the instance to <see cref="TensorEngine.Linalg"/> of the cached engine.</para>
    ///     <para><b>Nested calls.</b> A member that needs another numpy.linalg function calls its public facade
    ///     (<c>np.linalg.inv</c>, <c>np.linalg.svdvals</c>, <c>np.linalg.norm</c>, ...) where NumPy's statement calls the
    ///     module function, so the engine assigned to <see cref="TensorEngine.Linalg"/> computes it and the class boundary
    ///     meets it where NumPy's would; one that calls a sibling directly in NumPy (pinv's and matrix_rank's <c>svd</c>,
    ///     the hermitian svd's <c>eigh</c>) calls this engine's own virtual member.</para>
    ///     <para>The statements were the internal operation class <c>LinalgEngine</c> (the np-layer dtype plan's P2) and
    ///     the kernels the ten <see cref="TensorEngine"/> members of the same names (0.70.0); phase PE's E2 made the class
    ///     this public sub-engine. The files sit beside the <c>np.linalg</c> facades rather than under <c>Backends/</c>:
    ///     the statements call np.* functions, which ratchet R2 counts against engine files. The kernels are
    ///     <c>Backends/Default/LinearAlgebra/DefaultLinalgEngine.Kernels.cs</c>.</para>
    /// </remarks>
    public partial class DefaultLinalgEngine : LinalgEngine
    {
        /// <summary>
        ///     The instance <see cref="TensorEngine.Linalg"/> returns while nothing else is assigned: shared by every engine,
        ///     because the class holds no state (each kernel reads the Blas of its operand's engine).
        /// </summary>
        internal static readonly DefaultLinalgEngine Shared = new();

        /// <summary>
        ///     Port of <c>_assert_stacked_2d</c> — a stack of matrices needs at least two axes.
        /// </summary>
        /// <param name="arrays">The operands, checked in order; the first failing one is reported.</param>
        /// <exception cref="LinAlgError">An operand of fewer than two axes: <c>{ndim}-dimensional array given. Array must
        ///     be at least two-dimensional</c>.</exception>
        internal static void AssertStacked2d(params NDArray[] arrays)
        {
            foreach (var a in arrays)
            {
                if (a.ndim < 2)
                    throw new LinAlgError(
                        $"{a.ndim}-dimensional array given. Array must be at least two-dimensional");
            }
        }

        /// <summary>
        ///     Port of <c>_assert_2d</c> — exactly two axes, no stacking (<c>lstsq</c> and <c>multi_dot</c>).
        /// </summary>
        /// <param name="arrays">The operands, checked in order; the first failing one is reported.</param>
        /// <exception cref="LinAlgError">An operand that is not two-dimensional: <c>{ndim}-dimensional array given. Array
        ///     must be two-dimensional</c>.</exception>
        internal static void Assert2d(params NDArray[] arrays)
        {
            foreach (var a in arrays)
            {
                if (a.ndim != 2)
                    throw new LinAlgError($"{a.ndim}-dimensional array given. Array must be two-dimensional");
            }
        }

        /// <summary>
        ///     Port of <c>_assert_stacked_square</c>. NumPy unpacks <c>a.shape[-2:]</c> into two
        ///     names, so a rank below 2 fails the UNPACK and reports the at-least-2-D message —
        ///     which is why this repeats that check rather than assuming a caller ran it first.
        /// </summary>
        /// <param name="arrays">The operands, checked in order; the first failing one is reported.</param>
        /// <exception cref="LinAlgError">An operand of fewer than two axes (the at-least-two-dimensional text), or one
        ///     whose last two axes differ (<c>Last 2 dimensions of the array must be square</c>).</exception>
        internal static void AssertStackedSquare(params NDArray[] arrays)
        {
            foreach (var a in arrays)
            {
                if (a.ndim < 2)
                    throw new LinAlgError(
                        $"{a.ndim}-dimensional array given. Array must be at least two-dimensional");

                if (a.Shape.dimensions[a.ndim - 2] != a.Shape.dimensions[a.ndim - 1])
                    throw new LinAlgError("Last 2 dimensions of the array must be square");
            }
        }

        /// <summary>
        ///     Port of <c>_assert_finite</c> — <c>eig</c>/<c>eigvals</c> reject an operand holding
        ///     inf or NaN (the general eigensolver has no defined answer for them), where
        ///     <c>eigh</c>/<c>eigvalsh</c> do NOT check.
        /// </summary>
        /// <remarks>
        ///     Only the float family can be non-finite; every integer/bool width — and NumSharp's
        ///     <c>Decimal</c>/<c>Char</c>, which have no inf/NaN representation — is finite by
        ///     construction, exactly as NumPy's <c>isfinite</c> is all-True for them. Skipping them
        ///     is equivalent to NumPy's check and avoids running <c>isfinite</c> on a dtype NumPy has
        ///     no loop for. Runs BEFORE <see cref="CommonType"/>, so a float16 NaN reports this
        ///     message rather than the "unsupported in linalg" one — matching NumPy's order.
        ///     <para>
        ///     NumPy's <c>isfinite(a).all()</c> is answered by the fused single-pass
        ///     <see cref="Backends.Kernels.FiniteScan.IsAllFinite"/> (the <c>asarray_chkfinite</c> kernel: the
        ///     same predicate per dtype, every layout) instead of a bool temp plus a reduction — on the small
        ///     matrices numpy.polynomial's <c>{p}roots</c> hands eigvals that temp was ~0.4 µs of a ~10 µs call.
        ///     </para>
        /// </remarks>
        /// <param name="arrays">The operands to check (any dtype, any layout).</param>
        /// <exception cref="LinAlgError"><c>Array must not contain infs or NaNs</c> — an operand holds ±inf or a NaN
        ///     (a complex one in either component).</exception>
        /// <exception cref="TypeError">An operand holds text: NumPy's <c>isfinite</c> has no string loop, raised whatever
        ///     the operand's size — for fixed-width strings the legacy resolver's <c>ufunc 'isfinite' not supported for the
        ///     input types, ...</c>, for <c>StringDType</c> the dispatcher's <c>ufunc 'isfinite' did not contain a loop with
        ///     signature matching types &lt;class 'numpy.dtypes.StringDType'&gt; -&gt; None</c>.</exception>
        internal static void AssertFinite(params NDArray[] arrays)
        {
            foreach (var a in arrays)
            {
                // NumPy's isfinite(a) resolves its loop before it reads an element: a string operand (even an empty
                // one) is refused by the legacy type resolver.
                // A StringDType operand puts a new-style DType into the dispatch key: the dispatcher names its class.
                if (Backends.Strings.NDStringUfuncs.IsVString(a))
                    throw Backends.Strings.NDStringUfuncs.VStringNoLoop("isfinite", new[] { Backends.Strings.NDStringUfuncs.UnaryDispatchClass(a) });
                if (Backends.Strings.NDStringUfuncs.IsString(a))
                    throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("isfinite");
                bool floaty = a.typecode is NPTypeCode.Half or NPTypeCode.Single
                    or NPTypeCode.Double or NPTypeCode.Complex;
                if (floaty && !Backends.Kernels.FiniteScan.IsAllFinite(a))
                    throw new LinAlgError("Array must not contain infs or NaNs");
            }
        }

        /// <summary>
        ///     The refusal NumPy's <c>_umath_linalg</c> gufunc raises when one of its operands holds fixed-width
        ///     strings: <c>_commonType</c> puts a string operand in its <c>rt = double</c> branch, so the gufunc runs
        ///     with a <c>d-&gt;d</c> (or, beside a complex operand, <c>D-&gt;D</c>) signature, and its input-cast check
        ///     refuses the text before LAPACK sees anything — an EMPTY string operand included.
        /// </summary>
        /// <param name="gufunc">The <c>_umath_linalg</c> gufunc NumPy calls (<c>"inv"</c>, <c>"cholesky_lo"</c>,
        ///     <c>"svd_f"</c>, <c>"solve1"</c>, ...).</param>
        /// <param name="operands">The gufunc's inputs, in its order; nothing happens when none holds strings.</param>
        /// <exception cref="ArgumentException">The first string operand: <c>Cannot cast ufunc 'inv' input from
        ///     dtype('&lt;U2') to dtype('float64') with casting rule 'same_kind'</c> (a multi-input gufunc names the
        ///     input: <c>input 1</c>). NumPy raises a UFuncTypeError; the house type of a ufunc input-cast refusal is
        ///     <see cref="ArgumentException"/>.</exception>
        /// <remarks>
        ///     Call it right after <see cref="CommonType"/> (whose "unsupported in linalg" refusal of float16 / decimal /
        ///     char comes first, as NumPy's <c>_commonType</c> runs before the gufunc) and before any core-dimension
        ///     check, which NumPy's gufunc makes after it resolves the loop.
        /// </remarks>
        internal static void ThrowIfTextGufunc(string gufunc, params NDArray[] operands)
        {
            int first = -1;
            bool complex = false;
            for (int i = 0; i < operands.Length; i++)
            {
                var op = operands[i];
                if (op is null)
                    continue;
                if (first < 0 && Backends.Strings.NDStringUfuncs.IsText(op))
                    first = i;
                if (op.typecode == NPTypeCode.Complex)
                    complex = true;
            }
            if (first < 0)
                return;
            // _commonType: any complex operand makes the loop complex128, a string or any other operand float64.
            string target = complex ? "complex128" : "float64";
            string input = operands.Length > 1 ? $"input {first}" : "input";
            throw new ArgumentException(
                $"Cannot cast ufunc '{gufunc}' {input} from {operands[first].dtype.ToString(true)} to dtype('{target}') " +
                "with casting rule 'same_kind'");
        }

        /// <summary>
        ///     Port of <c>_commonType</c> — the dtype a LAPACK route produces.
        /// </summary>
        /// <param name="arrays">The operands whose dtypes decide the loop.</param>
        /// <returns><see cref="NPTypeCode.Complex"/> when any operand is complex, else <see cref="NPTypeCode.Double"/>
        ///     when any operand is float64, bool, an integer or text, else <see cref="NPTypeCode.Single"/> (every operand
        ///     float32).</returns>
        /// <exception cref="TypeError">An operand of float16, decimal or char: <c>array type float16 is unsupported in
        ///     linalg</c> (the dtype's name in place of float16).</exception>
        /// <remarks>
        ///     NumPy's rule, probed against 2.4.2: bool and every integer widen to
        ///     <c>float64</c>; <c>float32</c> and <c>float64</c> keep their width; complex stays
        ///     complex. <c>float16</c> is REJECTED rather than widened
        ///     ("array type float16 is unsupported in linalg"), and NumSharp's two dtypes with no
        ///     NumPy counterpart — <c>Decimal</c> and <c>Char</c> — take the same exit, since a
        ///     LAPACK routine has no loop for them either.
        /// </remarks>
        internal static NPTypeCode CommonType(params NDArray[] arrays)
        {
            bool isComplex = false;
            bool wideningToDouble = false;

            foreach (var a in arrays)
            {
                switch (a.typecode)
                {
                    case NPTypeCode.Complex:
                        isComplex = true;
                        break;
                    case NPTypeCode.Single:
                        break;
                    case NPTypeCode.Double:
                        wideningToDouble = true;
                        break;
                    case NPTypeCode.Half:
                    case NPTypeCode.Decimal:
                    case NPTypeCode.Char:
                        throw new TypeError($"array type {NumpyName(a.typecode)} is unsupported in linalg");
                    default:
                        // bool, every integer width and the string classes: NumPy's `rt = double` branch for any
                        // non-inexact type (a string operand is then refused by the gufunc's input cast,
                        // ThrowIfTextGufunc).
                        wideningToDouble = true;
                        break;
                }
            }

            if (isComplex)
                return NPTypeCode.Complex;

            return wideningToDouble ? NPTypeCode.Double : NPTypeCode.Single;
        }

        /// <summary>The name NumPy prints for a dtype in the "unsupported in linalg" rejection.</summary>
        /// <param name="typeCode">The refused dtype (float16, decimal or char).</param>
        /// <returns><c>float16</c>, <c>decimal</c>, <c>char</c>, or the lower-cased type code for any other.</returns>
        private static string NumpyName(NPTypeCode typeCode)
            => typeCode switch
            {
                NPTypeCode.Half => "float16",
                NPTypeCode.Decimal => "decimal",
                NPTypeCode.Char => "char",
                _ => typeCode.ToString().ToLowerInvariant()
            };

        /// <summary>
        ///     Casts an operand to the dtype the factorisation runs at. Validation happens before
        ///     this, so a rejected dtype never reaches a cast.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="common">The loop dtype (<see cref="CommonType"/>'s answer).</param>
        /// <returns><paramref name="a"/> itself when it already has the dtype, else a cast copy (a temporary the calling
        ///     <c>[NDScoped]</c> body reclaims).</returns>
        internal static NDArray ToCommon(NDArray a, NPTypeCode common)
            => a.typecode == common ? a : a.TensorEngine.Cast(a, common.AsType(), copy: false);

        /// <summary>
        ///     Element-wise <c>|x|²</c> as a REAL array — NumPy's <c>(x.conj() * x).real</c>, rounded as NumPy rounds it.
        /// </summary>
        /// <param name="x">The operand (a float or complex dtype, or decimal: norm converts the others first).</param>
        /// <returns>A fresh array: for a complex operand the float64 real part of NumPy's complex product
        ///     <c>conj(x) * x</c>, <c>fma(re, re, round(im * im))</c>; otherwise <c>x * x</c> in the operand's dtype.</returns>
        /// <remarks>
        ///     <para>
        ///     Spelled out rather than composed from <c>abs</c> because for a complex operand
        ///     <c>abs(z)² = hypot(re, im)²</c> is NOT <c>re² + im²</c>: the intermediate square
        ///     root rounds, so a Frobenius norm built that way drifts from NumPy's.
        ///     </para>
        ///     <para>
        ///     NumPy's complex multiply is <c>simd_cmul</c>: the real part is <c>muladdsub(a_re, b_re, a_im * b_im)</c>,
        ///     one fused multiply-subtract over a rounded product. With <c>a = conj(x)</c> and <c>b = x</c> that is
        ///     <c>re * re - (-im * im)</c>: the square of the real part fused with the rounded square of the imaginary part,
        ///     one rounding of the sum where <c>re * re + im * im</c> rounds twice (probed on 20,000 random values: the
        ///     two-rounding sum differs on 16 % of them, the fused one on none, contiguous or strided). A real operand's
        ///     <c>x.conj()</c> is a copy of it, so the product is <c>x * x</c>, which keeps a NaN's sign where
        ///     <c>|x| * |x|</c> cleared it.
        ///     </para>
        ///     <para>
        ///     Not modelled: which NaN the complex product returns when its operands hold several (the MSVC-compiled
        ///     <c>simd_cmul</c>'s operand order decides it), so a complex NaN's sign or payload can differ.
        ///     </para>
        /// </remarks>
        [NDScopedCovered] // only callers: the norm helpers, themselves covered under the [NDScoped] norm entries
        internal static NDArray AbsSquared(NDArray x)
        {
            if (x.typecode != NPTypeCode.Complex)
                return np.multiply(x, x);

            var magnitudes = np.zeros(Shape.Adopt(x.Shape.dimensions), NPTypeCode.Double);

            // Both walks are pinned to 'C' — the typed iterator defaults to 'K' (MEMORY order),
            // which for a strided or transposed source visits elements in a different sequence
            // than it does for the freshly allocated C-contiguous destination.
            // The writer is driven by hand, so nothing disposes it but this `using`: its enumerator owns a native
            // iterator state block (NDIterRef), which leaked on every complex call while the enumerator was a plain local.
            using var writer = np.nditer<double>(magnitudes, writeable: true, order: 'C').GetEnumerator();
            foreach (Complex z in np.nditer<Complex>(x, order: 'C'))
            {
                writer.MoveNext();
                // simd_cmul's real lane for conj(x) * x: re * re fused with the rounded im * im (remarks).
                writer.Current = Math.FusedMultiplyAdd(z.Real, z.Real, z.Imaginary * z.Imaginary);
            }

            return magnitudes;
        }
    }
}
