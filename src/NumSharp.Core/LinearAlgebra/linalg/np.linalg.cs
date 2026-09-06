using System;
using System.Numerics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>numpy.linalg</c> module.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <b>Most of this module needs a matrix backend and raises
        ///     <see cref="OpenBlasMissingBackendException"/> without one.</b> NumSharp.Core is 100 % managed C#
        ///     and carries no LU, QR, SVD or eigensolver, so — unlike the matrix products, which
        ///     always have a managed kernel to fall back on — a factorisation has nothing to compute
        ///     with until something assigns <see cref="TensorEngine.Blas"/>. The validation in front
        ///     of the throw is NumPy-exact, so shape, rank and argument errors are the real ones.
        ///     </para>
        ///     <para>
        ///     <b>What works today</b>, because it composes out of primitives NumSharp already has:
        ///     <see cref="multi_dot"/>, <see cref="matrix_power"/> at a NON-NEGATIVE exponent, the
        ///     Array-API forms <see cref="matmul"/>/<see cref="outer"/>/<see cref="tensordot"/>/
        ///     <see cref="trace"/>/<see cref="diagonal"/>/<see cref="cross"/>/
        ///     <see cref="matrix_transpose"/>/<see cref="vecdot"/>, and every
        ///     <see cref="norm"/> order except the three that are defined through singular values
        ///     (matrix <c>ord</c> 2, -2 and 'nuc').
        ///     </para>
        ///     <para>
        ///     <b>The Array-API forms are NOT aliases of their main-namespace twins</b> — NumPy gives
        ///     them different signatures and different defaults, and the differences are observable:
        ///     <c>np.linalg.trace</c> reduces the LAST two axes where <c>np.trace</c> reduces the
        ///     first two, <c>np.linalg.outer</c> demands 1-D operands where <c>np.outer</c> flattens
        ///     anything, and <c>np.linalg.cross</c> requires 3-vectors where <c>np.cross</c> still
        ///     accepts 2-vectors. Each is implemented against its own contract.
        ///     </para>
        /// </remarks>
        [ModuleName("np.linalg")]
        public static partial class linalg
        {
            /// <summary>
            ///     Port of <c>_assert_stacked_2d</c> — a stack of matrices needs at least two axes.
            /// </summary>
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
            ///     Port of <c>_assert_2d</c> — exactly two axes, no stacking (<c>lstsq</c> only).
            /// </summary>
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
            internal static void AssertStackedSquare(params NDArray[] arrays)
            {
                foreach (var a in arrays)
                {
                    if (a.ndim < 2)
                        throw new LinAlgError(
                            $"{a.ndim}-dimensional array given. Array must be at least two-dimensional");

                    if (a.shape[a.ndim - 2] != a.shape[a.ndim - 1])
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
            /// </remarks>
            internal static void AssertFinite(params NDArray[] arrays)
            {
                foreach (var a in arrays)
                {
                    bool floaty = a.typecode is NPTypeCode.Half or NPTypeCode.Single
                        or NPTypeCode.Double or NPTypeCode.Complex;
                    if (floaty && !np.all(np.isfinite(a)))
                        throw new LinAlgError("Array must not contain infs or NaNs");
                }
            }

            /// <summary>
            ///     Port of <c>_commonType</c> — the dtype a LAPACK route produces.
            /// </summary>
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
                            // bool and every integer width: NumPy's `result_type = double` branch.
                            wideningToDouble = true;
                            break;
                    }
                }

                if (isComplex)
                    return NPTypeCode.Complex;

                return wideningToDouble ? NPTypeCode.Double : NPTypeCode.Single;
            }

            /// <summary>The name NumPy prints for a dtype in the "unsupported in linalg" rejection.</summary>
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
            internal static NDArray ToCommon(NDArray a, NPTypeCode common)
                => a.typecode == common ? a : a.TensorEngine.Cast(a, common.AsType(), copy: false);

            /// <summary>
            ///     Element-wise <c>|x|²</c> as a REAL array — NumPy's <c>(x.conj() * x).real</c>.
            /// </summary>
            /// <remarks>
            ///     Spelled out rather than composed from <c>abs</c> because for a complex operand
            ///     <c>abs(z)² = hypot(re, im)²</c> is NOT <c>re² + im²</c>: the intermediate square
            ///     root rounds, so a Frobenius norm built that way drifts from NumPy's. For every
            ///     real dtype this is the ordinary square.
            /// </remarks>
            [NDScopedCovered] // only callers: the norm helpers, themselves covered under the [NDScoped] norm entries
            internal static NDArray AbsSquared(NDArray x)
            {
                if (x.typecode != NPTypeCode.Complex)
                {
                    var real = np.abs(x);
                    return np.multiply(real, real);
                }

                var magnitudes = np.zeros(new Shape(x.shape), NPTypeCode.Double);

                // Both walks are pinned to 'C' — the typed iterator defaults to 'K' (MEMORY order),
                // which for a strided or transposed source visits elements in a different sequence
                // than it does for the freshly allocated C-contiguous destination.
                var writer = np.nditer<double>(magnitudes, writeable: true, order: 'C').GetEnumerator();
                foreach (Complex z in np.nditer<Complex>(x, order: 'C'))
                {
                    writer.MoveNext();
                    writer.Current = z.Real * z.Real + z.Imaginary * z.Imaginary;
                }

                return magnitudes;
            }
        }
    }
}
