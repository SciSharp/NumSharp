using System;

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
        ///     of the throw is NumPy-exact, so shape, rank and argument errors are the real ones. An
        ///     EMPTY operand (an empty matrix or a zero-size batch) is the exception: NumPy's answer for it
        ///     needs no arithmetic (shapes, zeros, and the identity that a full SVD or a complete QR of an
        ///     empty matrix carries), so every factorisation gives it without a backend
        ///     (<see cref="Backends.LinAlgEmpty"/>).
        ///     </para>
        ///     <para>
        ///     <b>What works today</b>, because it composes out of primitives NumSharp already has:
        ///     <see cref="multi_dot(NDArray[], NDArray)"/>, <see cref="matrix_power"/> at a NON-NEGATIVE exponent, the
        ///     Array-API forms <see cref="matmul"/>/<see cref="outer"/>/<see cref="tensordot(NDArray, NDArray, int)"/>/
        ///     <see cref="trace"/>/<see cref="diagonal"/>/<see cref="cross"/>/
        ///     <see cref="matrix_transpose"/>/<see cref="vecdot"/>, and every
        ///     <see cref="norm(NDArray, object, int?, bool)"/> order except the three that are defined through singular values
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
        ///     <para>
        ///     <b>The bodies are the <see cref="TensorEngine.Linalg"/> sub-engine</b> (<see cref="LinalgEngine"/>, by default
        ///     <see cref="Backends.DefaultLinalgEngine"/> in <c>Backends/Default/LinearAlgebra/DefaultLinalgEngine*.cs</c>): NumPy's
        ///     <c>_linalg.py</c> statements (the argument checks, the dtype rule, the text refusals) around kernels that
        ///     stand for NumPy's <c>_umath_linalg</c> gufuncs. A facade here keeps its signature, its documentation, the
        ///     class boundary (numpy.matrix, numpy.char.chararray), a null check of the operand and one call,
        ///     <c>a.TensorEngine.Linalg.X(...)</c>, so an engine assigned to <see cref="TensorEngine.Linalg"/> computes every
        ///     function of this module.
        ///     </para>
        /// </remarks>
        [ModuleName("np.linalg")]
        public static partial class linalg
        {
        }
    }
}
