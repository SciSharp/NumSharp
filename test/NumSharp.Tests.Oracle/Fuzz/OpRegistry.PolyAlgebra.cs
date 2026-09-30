using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> series algebra (plan U2): <c>{p}mulx</c>, <c>{p}mul</c>, <c>{p}div</c> (a tuple
    ///     case), <c>{p}pow</c>, <c>{p}fromroots</c> for the six bases and <c>X2poly</c> / <c>poly2X</c> for the five
    ///     non-power ones. Pairs 1:1 with <c>gen_oracle.py</c>'s <c>gen_polyalgebra</c>, which writes the portable
    ///     <c>polyalgebra.jsonl</c> and the host-pinned <c>polyalgebra_parity.jsonl</c> (the BLAS-bound products) — the
    ///     same keys, dispatched here either way.
    ///
    ///     <para>
    ///     <b>Arguments are named</b> and decoded in the generator's fixed order — <c>c</c>, <c>c1</c>, <c>c2</c>,
    ///     <c>roots</c>, <c>pol</c>, <c>pow</c> — so operands are consumed as they were encoded; each is the polyseries
    ///     encoding (an operand, a Python scalar / list / tuple / str), which the facades take as <c>object</c> exactly as
    ///     NumPy takes an array_like. <c>maxpower</c> is a plain JSON int, or JSON null for an explicit <c>None</c>;
    ///     absent means the facade's own default (NumPy's: <c>None</c> for <c>polypow</c>, 16 for the others), so the
    ///     call omits it the way a port that omits it does.
    ///     </para>
    ///
    ///     <para>
    ///     <b>The pow argument binds like C# source.</b> A Python int that fits an <see cref="int"/> binds the facade's
    ///     <c>int</c> overload (the literal a port writes); a Python float, or an int past <see cref="int"/>'s range,
    ///     binds the <c>double</c> overload — where a C# <see cref="long"/> argument lands too (long converts to double
    ///     implicitly, not to int). A <c>"facet": "flags"</c> case records each result's
    ///     <c>[C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA]</c> (<see cref="PolyArgs.CalcFacet"/>).
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Dispatches a series-algebra key to the facade member NumPy's name refers to, on the submodule the key's
        ///     module part names.
        /// </summary>
        /// <param name="module">The corpus module (<c>polynomial</c> … <c>hermite_e</c>).</param>
        /// <param name="name">The function name with the basis prefix removed (<c>mulx</c>, <c>mul</c>, <c>pow</c>,
        ///     <c>fromroots</c>, <c>2poly</c>), or the whole name for a <c>poly2X</c> conversion.</param>
        /// <param name="a">The case's argument decoder.</param>
        /// <param name="result">The facade's result, or its flags facet.</param>
        /// <returns>True when <paramref name="name"/> is a series-algebra function; false leaves the key to the additive
        ///     family's dispatch.</returns>
        /// <exception cref="NotSupportedException">An unknown polynomial module.</exception>
        private static bool TryApplyPolyAlgebra(string module, string name, PolyArgs a, out NDArray result)
        {
            // Every arm calls the facade member of the SAME name NumPy's key names; a non-U2 name yields null and falls
            // back to the additive / calculus dispatch in ApplyPolySeries.
            switch (module)
            {
                case "polynomial":
                {
                    var m = np.polynomial.polynomial;
                    result = name switch
                    {
                        "mulx" => m.polymulx(a.Get("c")),
                        "mul" => m.polymul(a.Get("c1"), a.Get("c2")),
                        "pow" => PolyPow(a, (c, k) => m.polypow(c, k), (c, k, x) => m.polypow(c, k, x),
                                         (c, d) => m.polypow(c, d), (c, d, x) => m.polypow(c, d, x)),
                        "fromroots" => m.polyfromroots(a.Get("roots")),
                        _ => null,
                    };
                    break;
                }
                case "chebyshev":
                {
                    var m = np.polynomial.chebyshev;
                    result = name switch
                    {
                        "mulx" => m.chebmulx(a.Get("c")),
                        "mul" => m.chebmul(a.Get("c1"), a.Get("c2")),
                        "pow" => PolyPow(a, (c, k) => m.chebpow(c, k), (c, k, x) => m.chebpow(c, k, x),
                                         (c, d) => m.chebpow(c, d), (c, d, x) => m.chebpow(c, d, x)),
                        "fromroots" => m.chebfromroots(a.Get("roots")),
                        "2poly" => m.cheb2poly(a.Get("c")),
                        "poly2cheb" => m.poly2cheb(a.Get("pol")),
                        _ => null,
                    };
                    break;
                }
                case "legendre":
                {
                    var m = np.polynomial.legendre;
                    result = name switch
                    {
                        "mulx" => m.legmulx(a.Get("c")),
                        "mul" => m.legmul(a.Get("c1"), a.Get("c2")),
                        "pow" => PolyPow(a, (c, k) => m.legpow(c, k), (c, k, x) => m.legpow(c, k, x),
                                         (c, d) => m.legpow(c, d), (c, d, x) => m.legpow(c, d, x)),
                        "fromroots" => m.legfromroots(a.Get("roots")),
                        "2poly" => m.leg2poly(a.Get("c")),
                        "poly2leg" => m.poly2leg(a.Get("pol")),
                        _ => null,
                    };
                    break;
                }
                case "laguerre":
                {
                    var m = np.polynomial.laguerre;
                    result = name switch
                    {
                        "mulx" => m.lagmulx(a.Get("c")),
                        "mul" => m.lagmul(a.Get("c1"), a.Get("c2")),
                        "pow" => PolyPow(a, (c, k) => m.lagpow(c, k), (c, k, x) => m.lagpow(c, k, x),
                                         (c, d) => m.lagpow(c, d), (c, d, x) => m.lagpow(c, d, x)),
                        "fromroots" => m.lagfromroots(a.Get("roots")),
                        "2poly" => m.lag2poly(a.Get("c")),
                        "poly2lag" => m.poly2lag(a.Get("pol")),
                        _ => null,
                    };
                    break;
                }
                case "hermite":
                {
                    var m = np.polynomial.hermite;
                    result = name switch
                    {
                        "mulx" => m.hermmulx(a.Get("c")),
                        "mul" => m.hermmul(a.Get("c1"), a.Get("c2")),
                        "pow" => PolyPow(a, (c, k) => m.hermpow(c, k), (c, k, x) => m.hermpow(c, k, x),
                                         (c, d) => m.hermpow(c, d), (c, d, x) => m.hermpow(c, d, x)),
                        "fromroots" => m.hermfromroots(a.Get("roots")),
                        "2poly" => m.herm2poly(a.Get("c")),
                        "poly2herm" => m.poly2herm(a.Get("pol")),
                        _ => null,
                    };
                    break;
                }
                case "hermite_e":
                {
                    var m = np.polynomial.hermite_e;
                    result = name switch
                    {
                        "mulx" => m.hermemulx(a.Get("c")),
                        "mul" => m.hermemul(a.Get("c1"), a.Get("c2")),
                        "pow" => PolyPow(a, (c, k) => m.hermepow(c, k), (c, k, x) => m.hermepow(c, k, x),
                                         (c, d) => m.hermepow(c, d), (c, d, x) => m.hermepow(c, d, x)),
                        "fromroots" => m.hermefromroots(a.Get("roots")),
                        "2poly" => m.herme2poly(a.Get("c")),
                        "poly2herme" => m.poly2herme(a.Get("pol")),
                        _ => null,
                    };
                    break;
                }
                default:
                    throw new NotSupportedException($"polynomial module '{module}' is not registered in OpRegistry");
            }
            if (result is null)
                return false;
            result = a.CalcFacet(result);
            return true;
        }

        /// <summary>
        ///     <c>{p}div</c>'s <c>(quo, rem)</c> as the corpus records it: both slots, each the array or — for a
        ///     <c>"facet": "flags"</c> case — its layout flags (NumPy's remainder is a trimseq VIEW in several branches).
        /// </summary>
        /// <param name="op">The full key (<c>legendre.legdiv</c>).</param>
        /// <param name="a">The case's argument decoder.</param>
        /// <returns>The two slots.</returns>
        /// <exception cref="NotSupportedException">A key that is not one of the six divisions.</exception>
        private static NDArray[] ApplyPolyAlgebraTuple(string op, PolyArgs a)
        {
            // c1 is decoded before c2 (C# evaluates arguments left to right), the generator's encode order.
            (NDArray quo, NDArray rem) r = op switch
            {
                "polynomial.polydiv" => np.polynomial.polynomial.polydiv(a.Get("c1"), a.Get("c2")),
                "chebyshev.chebdiv" => np.polynomial.chebyshev.chebdiv(a.Get("c1"), a.Get("c2")),
                "legendre.legdiv" => np.polynomial.legendre.legdiv(a.Get("c1"), a.Get("c2")),
                "laguerre.lagdiv" => np.polynomial.laguerre.lagdiv(a.Get("c1"), a.Get("c2")),
                "hermite.hermdiv" => np.polynomial.hermite.hermdiv(a.Get("c1"), a.Get("c2")),
                "hermite_e.hermediv" => np.polynomial.hermite_e.hermediv(a.Get("c1"), a.Get("c2")),
                _ => throw new NotSupportedException($"polynomial tuple op '{op}' is not registered in OpRegistry"),
            };
            return new[] { a.CalcFacet(r.quo), a.CalcFacet(r.rem) };
        }

        /// <summary>
        ///     Calls a <c>{p}pow</c> facade the way C# source spelling the case would bind it: the <c>int</c> overload for
        ///     a Python int within <see cref="int"/>'s range, the <c>double</c> overload for a Python float or a larger
        ///     int (a C# <see cref="long"/> converts implicitly to <see cref="double"/>, never to <see cref="int"/>), with
        ///     <c>maxpower</c> passed only when the case gives it (JSON null = an explicit <c>None</c>).
        /// </summary>
        /// <param name="a">The case's argument decoder (<c>c</c>, <c>pow</c>, optional <c>maxpower</c>).</param>
        /// <param name="intDefault">The int overload at the facade's default maxpower.</param>
        /// <param name="intMax">The int overload with an explicit maxpower.</param>
        /// <param name="dblDefault">The double overload at the facade's default maxpower.</param>
        /// <param name="dblMax">The double overload with an explicit maxpower.</param>
        /// <returns>The facade's result.</returns>
        /// <exception cref="NotSupportedException">A pow spec that is not a Python int or float (C# has no overload for
        ///     it, so no port can spell the call).</exception>
        private static NDArray PolyPow(PolyArgs a,
                                       Func<object, int, NDArray> intDefault, Func<object, int, int?, NDArray> intMax,
                                       Func<object, double, NDArray> dblDefault, Func<object, double, int?, NDArray> dblMax)
        {
            // Operand order: the series first (it may consume an operand), then the scalar power.
            object c = a.Get("c");
            object pow = a.Get("pow");
            bool hasMax = a.Has("maxpower");
            int? maxpower = hasMax ? a.NullableInt("maxpower") : null;
            if (pow is long l && l >= int.MinValue && l <= int.MaxValue)
                return hasMax ? intMax(c, (int)l, maxpower) : intDefault(c, (int)l);
            double d = pow switch
            {
                long v => v,
                ulong v => v,
                BigInteger v => (double)v,
                double v => v,
                _ => throw new NotSupportedException($"pow of type {pow?.GetType().Name ?? "null"} has no {nameof(PolyPow)} overload"),
            };
            return hasMax ? dblMax(c, d, maxpower) : dblDefault(c, d);
        }
    }
}
