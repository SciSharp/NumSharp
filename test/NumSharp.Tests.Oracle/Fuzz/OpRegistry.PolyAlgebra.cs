using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;
using NumSharp;
using NumSharp.Backends.Kernels;

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
    ///     encoding (an operand, a Python scalar / list / tuple / str / None, np.float16), which the facades take as
    ///     <c>object</c> exactly as NumPy takes an array_like. <c>maxpower</c> is a plain JSON int, JSON null for an
    ///     explicit <c>None</c>, or a spec for any other kind; absent means the facade's own default (NumPy's: <c>None</c>
    ///     for <c>polypow</c>, 16 for the others), so the call omits it the way a port that omits it does.
    ///     </para>
    ///
    ///     <para>
    ///     <b>The pow argument binds like C# source.</b> A Python int that fits an <see cref="int"/> binds the facade's
    ///     <c>int</c> overload (the literal a port writes); a Python float, or an int past <see cref="int"/>'s range that a
    ///     <see cref="long"/> / <see cref="ulong"/> holds, binds the <c>double</c> overload (they convert to double
    ///     implicitly, not to int); every other power — a bool, a BigInteger, np.float16, a complex, a str, None, a list /
    ///     tuple, an ndarray — and any limit that is not a plain int binds the <c>object</c> overloads. A
    ///     <c>"facet": "flags"</c> case records each result's <c>[C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA]</c>
    ///     (<see cref="PolyArgs.CalcFacet"/>).
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
                        "pow" => PolyPow(a, PolyBasis.Power, (c, k) => m.polypow(c, k), (c, k, x) => m.polypow(c, k, x)),
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
                        "pow" => PolyPow(a, PolyBasis.Chebyshev, (c, k) => m.chebpow(c, k), (c, k, x) => m.chebpow(c, k, x)),
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
                        "pow" => PolyPow(a, PolyBasis.Legendre, (c, k) => m.legpow(c, k), (c, k, x) => m.legpow(c, k, x)),
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
                        "pow" => PolyPow(a, PolyBasis.Laguerre, (c, k) => m.lagpow(c, k), (c, k, x) => m.lagpow(c, k, x)),
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
                        "pow" => PolyPow(a, PolyBasis.Hermite, (c, k) => m.hermpow(c, k), (c, k, x) => m.hermpow(c, k, x)),
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
                        "pow" => PolyPow(a, PolyBasis.HermiteE, (c, k) => m.hermepow(c, k), (c, k, x) => m.hermepow(c, k, x)),
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
        ///     Replays a <c>{p}pow</c> case the way C# source can spell it. The facade's one signature,
        ///     <c>(c, int pow, int? maxpower)</c> (API-parity batch 11, the owner's decision 2: typed parameters only), takes
        ///     a Python int within <see cref="int"/>'s range with <c>maxpower</c> absent, None or a plain int. Every other
        ///     power or limit - a float, a bool, a BigInteger, np.float16, a complex, a str, None, a list / tuple, an ndarray,
        ///     a Python int past int's range, a spec-encoded limit - has no C# spelling since then, so the case replays
        ///     through the computation the removed object facades called, <see cref="NDPolyAlgebra.Pow(PolyBasis, object,
        ///     object, object)"/>: NumPy's <c>int(pow)</c> and <c>power &gt; maxpower</c> over any value, the machinery the
        ///     series class's <c>**</c> still reaches. The corpus stays whole.
        /// </summary>
        /// <param name="a">The case's argument decoder (<c>c</c>, <c>pow</c>, optional <c>maxpower</c>: a raw JSON int or
        ///     null, or a spec for any other kind).</param>
        /// <param name="basis">The case's basis (the computation's first argument).</param>
        /// <param name="intDefault">The facade at its default maxpower.</param>
        /// <param name="intMax">The facade with an explicit maxpower.</param>
        /// <returns>The facade's or the computation's result.</returns>
        private static NDArray PolyPow(PolyArgs a, PolyBasis basis,
                                       Func<object, int, NDArray> intDefault, Func<object, int, int?, NDArray> intMax)
        {
            // Operand order: the series first (it may consume an operand), then the power, then the limit.
            object c = a.Get("c");
            object pow = a.Get("pow");
            bool hasMax = a.Has("maxpower");
            // A plain JSON int / null is what the facade's int? parameter takes; any spec-encoded limit (a float, a bool, a
            // str, an ndarray, a Python int past int's range, ...) has no C# spelling.
            bool plainMax = !hasMax || a.Raw("maxpower").ValueKind is JsonValueKind.Null or JsonValueKind.Number;
            if (plainMax && pow is long l && l >= int.MinValue && l <= int.MaxValue)
            {
                int? maxpower = hasMax ? a.NullableInt("maxpower") : null;
                return hasMax ? intMax(c, (int)l, maxpower) : intDefault(c, (int)l);
            }

            // The facades' defaults: polypow has no limit (None), every other basis 16 (the boxed Python int the removed
            // object facades passed).
            object limit = hasMax
                ? (plainMax ? (object)a.NullableInt("maxpower") : a.Get("maxpower"))
                : basis == PolyBasis.Power ? null : NDPolyAlgebra.DefaultMaxPower;
            // The removed facades were [NDScoped]: the scope reclaims the computation's intermediates (the scope-audit
            // sweep replays through this registry and would count them as undisposed) and hands the result back.
            using var scope = NDScope.Open();
            var result = NDPolyAlgebra.Pow(basis, c, pow, limit);
            scope.Returns(result);
            return result;
        }
    }
}
