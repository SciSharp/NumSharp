using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.qr</c> past the mode check and the class boundary: <c>_assert_stacked_2d</c>,
        ///     <c>_commonType</c>, then <see cref="TensorEngine.Qr"/> at the loop dtype; the raw and economic modes' factored
        ///     matrix comes back as NumPy's object (<see cref="RawFactor"/>, <see cref="FactoredCopy"/>).
        /// </summary>
        /// <param name="a">The matrix, or a stack of them: plain for a numpy.matrix operand (the facade applied its rule), a
        ///     numpy.char.chararray one as it is (NumPy's <c>_makearray</c> drops that class; the results are plain).</param>
        /// <param name="resolved">The mode <see cref="ResolveQrMode"/> returned.</param>
        /// <returns><c>(Q, R)</c>; for <c>"r"</c> the Q slot is null, for <c>"raw"</c> the pair is <c>(h, tau)</c>, for
        ///     <c>"economic"</c> the packed matrix and null.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped] // reclaims the ToCommon cast temp and the backend's C-ordered factor; the (Q, R) tuple is yielded
        internal static (NDArray Q, NDArray R) Qr(NDArray a, string resolved)
        {
            AssertStacked2d(a);
            var common = CommonType(a);
            var (q, r) = a.TensorEngine.Qr(ToCommon(a, common), resolved);
            // raw and economic hand back the factored matrix itself, and NumPy's is its copy of the operand
            // (`a = a.astype(t, copy=True)`, the operand's K order) that qr_r_raw factored in place.
            return resolved switch
            {
                "raw" => (RawFactor(a, q), r),
                "economic" => (FactoredCopy(a, q), r),
                _ => (q, r),
            };
        }

        /// <summary>
        ///     NumPy's raw-mode <c>q = transpose(a); q = q.astype(result_t, copy=False)</c>: a transposed view of the
        ///     factored copy (<see cref="FactoredCopy"/>), or for a float32 operand (factored in float64) the cast's new
        ///     array in that view's order.
        /// </summary>
        /// <param name="a">The operand as the caller passed it: its memory order decides the copy's.</param>
        /// <param name="h">The backend's reflectors, shape <c>(..., N, M)</c>, C-ordered, of the result dtype.</param>
        /// <returns>h's values: for float64 / complex128 a view of a new <c>(..., M, N)</c> array in <paramref name="a"/>'s
        ///     K order (F-contiguous for a C operand, C-contiguous for an F one, not an owner), for float32 a new array in
        ///     that view's order.</returns>
        [NDScopedCovered] // only caller: [NDScoped] Qr; the factored copy behind the returned view and h's transposed view ride its scope
        private static NDArray RawFactor(NDArray a, NDArray h)
        {
            var q = np.swapaxes(FactoredCopy(a, np.swapaxes(h, -1, -2)), -1, -2);
            // t differs from result_t only for float32 (computed in float64): astype(float32, copy=False) of the float64
            // view is then a copy, in the view's K order.
            return q.typecode == NPTypeCode.Single ? q.copy('K') : q;
        }

        /// <summary>
        ///     NumPy's <c>a = a.astype(t, copy=True)</c> after <c>qr_r_raw</c> wrote the factorisation into it: a new plain
        ///     array of <paramref name="a"/>'s shape and K order holding <paramref name="factored"/>'s values. Economic mode
        ///     returns it as it is (its <c>a.astype(result_t, copy=False)</c> keeps the order when it casts a float32 operand
        ///     back).
        /// </summary>
        /// <param name="a">The operand as the caller passed it: its memory order decides the copy's. A numpy.char.chararray
        ///     holding numbers arrives as it is (the facade has no chararray rule), and the copy is plain all the same, as
        ///     NumPy's is (its <c>_makearray</c> took <c>asarray(a)</c>).</param>
        /// <param name="factored">The packed factorisation, shape <c>(..., M, N)</c>, of the result dtype.</param>
        /// <returns>The new array, owning its data, never a chararray or a matrix.</returns>
        private static NDArray FactoredCopy(NDArray a, NDArray factored)
        {
            // A plain region: np.empty_like must not apply its chararray rule to a chararray operand from inside the
            // library (it wrapped the float64 copy as a chararray: "Can only create a chararray from string data."). The
            // region does nothing for a plain operand, and a matrix one reaches here plain already (the facade's rule).
            using var plain = NDMatrixBoundary.PlainFor(a);
            var copy = np.empty_like(a, factored.dtype);
            np.copyto(copy, factored);
            return copy;
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
