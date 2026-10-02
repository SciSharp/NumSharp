using System;
using NumSharp;

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
        ///     <c>{p}vander(x, deg)</c> through the overload a C# caller binds: a Python int that fits <see cref="int"/> takes the
        ///     typed <c>(object x, int deg)</c> overload (what <c>polyvander(x, 3)</c> compiles to); every other degree kind — a
        ///     wider or huge int, bool, float, complex, str, None, a list / tuple, an operand array — the <c>object</c> overload,
        ///     whose <c>operator.index</c> port decides it. Both overloads are thereby replayed against NumPy.
        /// </summary>
        /// <param name="a">The case's argument decoder (x decoded first, then deg — the generator's order).</param>
        /// <param name="viaInt">The basis's <c>(object, int)</c> overload.</param>
        /// <param name="viaObject">The basis's <c>(object, object)</c> overload.</param>
        /// <returns>The facade's result.</returns>
        private static NDArray PolyVander1(PolyArgs a, Func<object, int, NDArray> viaInt, Func<object, object, NDArray> viaObject)
        {
            object x = a.Get("x");
            object deg = a.Get("deg");
            return deg is long l && l >= int.MinValue && l <= int.MaxValue ? viaInt(x, (int)l) : viaObject(x, deg);
        }
    }
}
