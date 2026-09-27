using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> half of the registry: the evaluation family of the six basis
    ///     submodules (<c>polynomial.polyval</c> … <c>hermite_e.hermegrid3d</c>, <c>*valnd</c>). Pairs 1:1 with
    ///     <c>gen_oracle.py</c>'s <c>gen_polyeval</c>.
    ///
    ///     <para>
    ///     <b>Why the keys are module-qualified.</b> The package reuses the legacy <c>np.polyval</c> family's
    ///     names with the OPPOSITE coefficient order (plan D5), so a bare <c>polyval</c> key would mix the two
    ///     corpora. The prefix names the submodule; the suffix is the NumPy function name verbatim.
    ///     </para>
    ///
    ///     <para>
    ///     <b>The x forms.</b> <c>params["xs"]</c> lists one entry per point argument, in order: <c>"a"</c>
    ///     consumes the next operand (an array — a 0-d one included, which NumPy treats as STRONG), a JSON
    ///     object is a Python scalar rebuilt as the C# primitive that means the same thing to the facade (a
    ///     WEAK value). The coefficients are always the last operand.
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>The corpus module prefixes of the polynomial package, as written by the generator.</summary>
        private static readonly string[] PolynomialPrefixes =
            { "polynomial.", "chebyshev.", "legendre.", "laguerre.", "hermite.", "hermite_e.", "polyutils." };

        /// <summary>Whether <paramref name="op"/> is a polynomial-package key.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <returns>True for the module-qualified polynomial keys.</returns>
        internal static bool IsPolynomialOp(string op)
        {
            foreach (var p in PolynomialPrefixes)
                if (op.StartsWith(p, StringComparison.Ordinal))
                    return true;
            return false;
        }

        /// <summary>
        ///     Dispatch a polynomial-package corpus op. The function kind (val / val2d / val3d / grid2d /
        ///     grid3d / valnd) is the name's suffix after the basis prefix; the submodule comes from the key's
        ///     module part, so the SAME facade member NumPy's name refers to is the one exercised.
        /// </summary>
        /// <param name="op">The full key (<c>chebyshev.chebval2d</c>).</param>
        /// <param name="p">The case's params (<c>xs</c>, optional <c>tensor</c>).</param>
        /// <param name="ops">Reconstructed operands: the array x/y/z/pts in order, then the coefficients.</param>
        /// <returns>The facade's result.</returns>
        /// <exception cref="NotSupportedException">An unknown module or function name.</exception>
        internal static NDArray ApplyPolynomial(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            int dot = op.IndexOf('.');
            string module = op.Substring(0, dot), fn = op.Substring(dot + 1);
            // The additive family and polyutils (U1, gen_polyseries) carry named arguments, not "xs".
            if (module == "polyutils" || !p.ContainsKey("xs"))
                return ApplyPolySeries(module, fn, p, ops);
            var xs = p["xs"];
            var args = new object[xs.GetArrayLength()];
            int next = 0;
            int i = 0;
            foreach (var e in xs.EnumerateArray())
                args[i++] = e.ValueKind == JsonValueKind.String ? ops[next++] : PythonScalar(e);
            NDArray c = ops[next];
            bool tensor = !p.TryGetValue("tensor", out var t) || t.GetBoolean();

            switch (module)
            {
                case "polynomial":
                {
                    var m = np.polynomial.polynomial;
                    return fn switch
                    {
                        "polyval" => m.polyval(args[0], c, tensor),
                        "polyval2d" => m.polyval2d((NDArray)args[0], (NDArray)args[1], c),
                        "polyval3d" => m.polyval3d((NDArray)args[0], (NDArray)args[1], (NDArray)args[2], c),
                        "polygrid2d" => m.polygrid2d(args[0], args[1], c),
                        "polygrid3d" => m.polygrid3d(args[0], args[1], args[2], c),
                        "polyvalnd" => m.polyvalnd(AsArrays(args), c),
                        _ => throw new NotSupportedException($"polynomial op '{op}' is not registered in OpRegistry"),
                    };
                }
                case "chebyshev":
                {
                    var m = np.polynomial.chebyshev;
                    return fn switch
                    {
                        "chebval" => m.chebval(args[0], c, tensor),
                        "chebval2d" => m.chebval2d((NDArray)args[0], (NDArray)args[1], c),
                        "chebval3d" => m.chebval3d((NDArray)args[0], (NDArray)args[1], (NDArray)args[2], c),
                        "chebgrid2d" => m.chebgrid2d(args[0], args[1], c),
                        "chebgrid3d" => m.chebgrid3d(args[0], args[1], args[2], c),
                        "chebvalnd" => m.chebvalnd(AsArrays(args), c),
                        _ => throw new NotSupportedException($"polynomial op '{op}' is not registered in OpRegistry"),
                    };
                }
                case "legendre":
                {
                    var m = np.polynomial.legendre;
                    return fn switch
                    {
                        "legval" => m.legval(args[0], c, tensor),
                        "legval2d" => m.legval2d((NDArray)args[0], (NDArray)args[1], c),
                        "legval3d" => m.legval3d((NDArray)args[0], (NDArray)args[1], (NDArray)args[2], c),
                        "leggrid2d" => m.leggrid2d(args[0], args[1], c),
                        "leggrid3d" => m.leggrid3d(args[0], args[1], args[2], c),
                        "legvalnd" => m.legvalnd(AsArrays(args), c),
                        _ => throw new NotSupportedException($"polynomial op '{op}' is not registered in OpRegistry"),
                    };
                }
                case "laguerre":
                {
                    var m = np.polynomial.laguerre;
                    return fn switch
                    {
                        "lagval" => m.lagval(args[0], c, tensor),
                        "lagval2d" => m.lagval2d((NDArray)args[0], (NDArray)args[1], c),
                        "lagval3d" => m.lagval3d((NDArray)args[0], (NDArray)args[1], (NDArray)args[2], c),
                        "laggrid2d" => m.laggrid2d(args[0], args[1], c),
                        "laggrid3d" => m.laggrid3d(args[0], args[1], args[2], c),
                        "lagvalnd" => m.lagvalnd(AsArrays(args), c),
                        _ => throw new NotSupportedException($"polynomial op '{op}' is not registered in OpRegistry"),
                    };
                }
                case "hermite":
                {
                    var m = np.polynomial.hermite;
                    return fn switch
                    {
                        "hermval" => m.hermval(args[0], c, tensor),
                        "hermval2d" => m.hermval2d((NDArray)args[0], (NDArray)args[1], c),
                        "hermval3d" => m.hermval3d((NDArray)args[0], (NDArray)args[1], (NDArray)args[2], c),
                        "hermgrid2d" => m.hermgrid2d(args[0], args[1], c),
                        "hermgrid3d" => m.hermgrid3d(args[0], args[1], args[2], c),
                        "hermvalnd" => m.hermvalnd(AsArrays(args), c),
                        _ => throw new NotSupportedException($"polynomial op '{op}' is not registered in OpRegistry"),
                    };
                }
                case "hermite_e":
                {
                    var m = np.polynomial.hermite_e;
                    return fn switch
                    {
                        "hermeval" => m.hermeval(args[0], c, tensor),
                        "hermeval2d" => m.hermeval2d((NDArray)args[0], (NDArray)args[1], c),
                        "hermeval3d" => m.hermeval3d((NDArray)args[0], (NDArray)args[1], (NDArray)args[2], c),
                        "hermegrid2d" => m.hermegrid2d(args[0], args[1], c),
                        "hermegrid3d" => m.hermegrid3d(args[0], args[1], args[2], c),
                        "hermevalnd" => m.hermevalnd(AsArrays(args), c),
                        _ => throw new NotSupportedException($"polynomial op '{op}' is not registered in OpRegistry"),
                    };
                }
                default:
                    throw new NotSupportedException($"polynomial module '{module}' is not registered in OpRegistry");
            }
        }

        /// <summary>The point arguments as arrays (valnd takes arrays only; the generator never emits a weak one).</summary>
        /// <param name="args">The rebuilt point arguments.</param><returns>The arrays.</returns>
        private static NDArray[] AsArrays(object[] args)
        {
            var r = new NDArray[args.Length];
            for (int i = 0; i < args.Length; i++) r[i] = (NDArray)args[i];
            return r;
        }

        /// <summary>
        ///     Rebuilds a Python scalar as the C# primitive the facade treats as that Python type: bool → bool,
        ///     int → long (ulong when it only fits there, <see cref="BigInteger"/> beyond — the polynomial facades
        ///     accept it as a Python int of any size), float → double (from its exact bit pattern, so NaN payloads
        ///     and -0.0 survive), complex → <see cref="Complex"/>.
        /// </summary>
        /// <param name="e">The generator's scalar spec.</param>
        /// <returns>The boxed primitive.</returns>
        /// <exception cref="NotSupportedException">An unknown kind.</exception>
        private static object PythonScalar(JsonElement e)
        {
            string kind = e.GetProperty("kind").GetString();
            switch (kind)
            {
                case "bool": return e.GetProperty("bool").GetBoolean();
                case "int":
                {
                    var bi = BigInteger.Parse(e.GetProperty("int").GetString(), CultureInfo.InvariantCulture);
                    if (bi >= long.MinValue && bi <= long.MaxValue) return (long)bi;
                    if (bi >= ulong.MinValue && bi <= ulong.MaxValue) return (ulong)bi;
                    return bi;
                }
                case "float": return Bits(e.GetProperty("bits").GetString());
                case "complex": return new Complex(Bits(e.GetProperty("re").GetString()), Bits(e.GetProperty("im").GetString()));
                default: throw new NotSupportedException($"unknown Python scalar kind '{kind}'");
            }

            static double Bits(string hex) =>
                BitConverter.Int64BitsToDouble(unchecked((long)ulong.Parse(hex.Substring(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture)));
        }
    }
}
