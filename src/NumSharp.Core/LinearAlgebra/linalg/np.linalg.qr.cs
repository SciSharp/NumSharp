using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     QR factorisation — <c>a = Q R</c> with Q orthonormal and R upper-triangular.
            /// </summary>
            /// <param name="a">The matrix, or a stack of them.</param>
            /// <param name="mode">
            ///     <c>"reduced"</c> (default) gives <c>Q:(...,M,K)</c>, <c>R:(...,K,N)</c> for
            ///     <c>K = min(M,N)</c>; <c>"complete"</c> gives a square <c>Q:(...,M,M)</c>;
            ///     <c>"r"</c> returns R alone; <c>"raw"</c> returns LAPACK's packed
            ///     <c>(h, tau)</c> pair rather than (Q, R).
            /// </param>
            /// <returns>
            ///     <c>(Q, R)</c>. For <c>"r"</c> the Q slot is null; for <c>"raw"</c> the pair is
            ///     <c>(h, tau)</c>.
            /// </returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>);
            ///     an unknown <paramref name="mode"/> (checked before the operand).</exception>
            /// <exception cref="LinAlgError">Fewer than two axes.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            [NDScoped] // what the call leaves (the Linalg member and the woven subclass path scope their own)
            public static (NDArray Q, NDArray R) qr(NDArray a, string mode = "reduced")
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                // NumPy checks the mode first. The member resolves it again (a resolved mode comes back unchanged), and so
                // does the subclass rule that decides which slots a matrix operand's class wraps (raw mode: h but not tau).
                Backends.DefaultLinalgEngine.ResolveQrMode(mode);
                return a.TensorEngine.Linalg.Qr(a, mode);
            }
        }
    }
}
