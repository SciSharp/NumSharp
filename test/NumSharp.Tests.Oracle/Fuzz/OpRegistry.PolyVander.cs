using System;
using NumSharp;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> Vandermonde family (plan U5): <c>{p}vander</c>, <c>{p}vander2d</c>, <c>{p}vander3d</c> of the
    ///     six bases. Pairs 1:1 with <c>gen_oracle.py</c>'s <c>gen_polyvander</c>: arguments are named (<c>x</c>, <c>y</c>,
    ///     <c>z</c>, <c>deg</c>), decoded in that order (operands consumed as the generator encoded them), each the same KIND
    ///     of value NumPy received — an operand, a Python scalar / list / tuple / str / None, or an np.float16 (Half).
    ///     A <c>"facet": "flags"</c> case records the result's <c>[C, F, OWNDATA]</c> instead of its values
    ///     (<see cref="PolyArgs.CalcFacet"/>).
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     <c>{p}vander(x, deg)</c> the way C# source can spell it. The facade's one signature, <c>(object x, int deg)</c>
        ///     (API-parity batch 11, the owner's decision 2: typed parameters only), takes a Python int that fits
        ///     <see cref="int"/> (what <c>polyvander(x, 3)</c> compiles to). Every other degree kind - a wider or huge int,
        ///     bool, float, complex, str, None, a list / tuple, an operand array - has no C# spelling since then, so the case
        ///     replays through the computation the removed object facade called
        ///     (<see cref="NDPolyVander.Vander(PolyBasis, object, object)"/>, whose <c>operator.index</c> port decides it, the
        ///     machinery vander2d / vander3d's per-axis degrees still reach). The corpus stays whole.
        /// </summary>
        /// <param name="a">The case's argument decoder (x decoded first, then deg — the generator's order).</param>
        /// <param name="basis">The case's basis (the computation's first argument).</param>
        /// <param name="viaInt">The basis's <c>(object, int)</c> facade.</param>
        /// <returns>The facade's or the computation's result.</returns>
        private static NDArray PolyVander1(PolyArgs a, PolyBasis basis, Func<object, int, NDArray> viaInt)
        {
            object x = a.Get("x");
            object deg = a.Get("deg");
            if (deg is long l && l >= int.MinValue && l <= int.MaxValue)
                return viaInt(x, (int)l);
            // The removed object facade was [NDScoped]: the scope reclaims the computation's intermediates (the scope-audit
            // sweep replays through this registry) and hands the matrix back.
            using var scope = NDScope.Open();
            var result = NDPolyVander.Vander(basis, x, deg);
            scope.Returns(result);
            return result;
        }
    }
}
