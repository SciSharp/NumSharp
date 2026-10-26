using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.qr</c> past the mode check and the class boundary: <c>_assert_stacked_2d</c>,
        ///     <c>_commonType</c>, then <see cref="TensorEngine.Qr"/> at the loop dtype.
        /// </summary>
        /// <param name="a">The matrix, or a stack of them (plain: the facade applied the class boundary).</param>
        /// <param name="resolved">The mode <see cref="ResolveQrMode"/> returned.</param>
        /// <returns><c>(Q, R)</c>; for <c>"r"</c> the Q slot is null, for <c>"raw"</c> the pair is <c>(h, tau)</c>, for
        ///     <c>"economic"</c> the packed matrix and null.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped] // reclaims the ToCommon cast temp; the (Q, R) tuple is yielded component-wise
        internal static (NDArray Q, NDArray R) Qr(NDArray a, string resolved)
        {
            AssertStacked2d(a);
            var common = CommonType(a);
            return a.TensorEngine.Qr(ToCommon(a, common), resolved);
        }

        /// <summary>
        ///     NumPy's mode parsing, including the deprecated one-letter aliases it still honours.
        /// </summary>
        /// <param name="mode">The caller's mode.</param>
        /// <returns><c>"reduced"</c>, <c>"complete"</c>, <c>"r"</c>, <c>"raw"</c> or <c>"economic"</c>.</returns>
        /// <exception cref="ValueError">A null or unknown mode: <c>Unrecognized mode '{mode}'</c>.</exception>
        internal static string ResolveQrMode(string mode)
        {
            if (mode is null)
                throw new ValueError("Unrecognized mode ''");

            switch (mode)
            {
                case "reduced":
                case "complete":
                case "r":
                case "raw":
                    return mode;
                // Kept from NumPy's own compatibility block: 'f'ull and 'e'conomic were the
                // pre-1.8 spellings and still map onto the modern modes.
                case "f":
                case "full":
                    return "reduced";
                case "e":
                case "economic":
                    return "economic";
                default:
                    throw new ValueError($"Unrecognized mode '{mode}'");
            }
        }
    }
}
