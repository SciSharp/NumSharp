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
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            [NDScoped] // reclaims the ToCommon cast temp; the (Q, R) tuple is yielded component-wise
            public static (NDArray Q, NDArray R) qr(NDArray a, string mode = "reduced")
            {
                string resolved = ResolveQrMode(mode);
                AssertStacked2d(a);
                var common = CommonType(a);
                return a.TensorEngine.Qr(ToCommon(a, common), resolved);
            }

            /// <summary>
            ///     NumPy's mode parsing, including the deprecated one-letter aliases it still honours.
            /// </summary>
            private static string ResolveQrMode(string mode)
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
}
