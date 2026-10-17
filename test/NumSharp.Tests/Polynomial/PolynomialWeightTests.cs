using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> weight functions (plan <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.2, unit
    ///     U8): <c>chebweight</c>, <c>legweight</c>, <c>lagweight</c>, <c>hermweight</c>, <c>hermeweight</c>. NumPy 2.4.2's own
    ///     <c>test_weight</c> cases ported; NumPy's byte dumps for every dtype rule, Python- and NumPy-scalar kind, memory
    ///     layout and error (the probe script that made them is quoted next to <see cref="Cases"/>); and what the oracle
    ///     (<c>polyweight.jsonl</c> / <c>polyweight_host.jsonl</c>) cannot see — string arrays, complex <c>chebweight</c>, decimal,
    ///     <c>numpy.matrix</c>, the two routes' agreement, and the CPython / scalarmath power rules behind <c>x**2</c>.
    /// </summary>
    /// <remarks>
    ///     The exp-based weights' float64, float16 and complex128 results come out of the C runtime's <c>exp</c> / <c>expf</c>
    ///     on both sides, and NumPy's dumps are win-amd64's (ucrtbase). On another host those values are held to 2 ULP of
    ///     NumPy's (<see cref="AssertCase"/>); everything else — <c>chebweight</c>, <c>legweight</c>, every float32 result
    ///     (NumPy's own SIMD exp, ported), dtypes, shapes, strides and errors — is asserted exactly on every host.
    /// </remarks>
    [TestClass]
    public class PolynomialWeightTests
    {
        /// <summary>The five weights by NumPy's prefix.</summary>
        private static readonly Dictionary<string, Func<object, NDArray>> Weights = new()
        {
            ["cheb"] = x => np.polynomial.chebyshev.chebweight(x),
            ["leg"] = x => np.polynomial.legendre.legweight(x),
            ["lag"] = x => np.polynomial.laguerre.lagweight(x),
            ["herm"] = x => np.polynomial.hermite.hermweight(x),
            ["herme"] = x => np.polynomial.hermite_e.hermeweight(x),
        };

        /// <summary>The basis of each weight prefix (the route-agreement test calls the engine directly).</summary>
        private static readonly Dictionary<string, PolyBasis> Bases = new()
        {
            ["cheb"] = PolyBasis.Chebyshev,
            ["leg"] = PolyBasis.Legendre,
            ["lag"] = PolyBasis.Laguerre,
            ["herm"] = PolyBasis.Hermite,
            ["herme"] = PolyBasis.HermiteE,
        };

        /// <summary>One NumPy 2.4.2 dump: the input, and either the result (dtype, shape, element strides, C-order bytes) or the
        ///     exception (Python type name and text).</summary>
        /// <param name="Weight">The weight's prefix.</param>
        /// <param name="Label">The dump's label (the probe's case name).</param>
        /// <param name="Input">Builds the input NumPy was given (the C# spelling of the same value kind).</param>
        /// <param name="Dtype">NumPy's result dtype, or null for an error case.</param>
        /// <param name="Shape">NumPy's result shape.</param>
        /// <param name="Strides">NumPy's result strides in elements (empty for 0-d and empty results: not compared).</param>
        /// <param name="Hex">NumPy's result bytes in C order.</param>
        /// <param name="ErrorType">The Python exception type of an error case.</param>
        /// <param name="ErrorText">NumPy's error text.</param>
        private sealed record Case(string Weight, string Label, Func<object> Input, string Dtype, long[] Shape, long[] Strides,
            string Hex, string ErrorType, string ErrorText);

        /// <summary>A dump with a result.</summary>
        /// <param name="w">The weight's prefix.</param>
        /// <param name="label">The dump's label.</param>
        /// <param name="input">Builds the input.</param>
        /// <param name="dtype">NumPy's result dtype.</param>
        /// <param name="shape">NumPy's result shape.</param>
        /// <param name="strides">NumPy's element strides (empty: not compared).</param>
        /// <param name="hex">NumPy's C-order bytes.</param>
        /// <returns>The case.</returns>
        private static Case Ok(string w, string label, Func<object> input, string dtype, long[] shape, long[] strides, string hex)
            => new(w, label, input, dtype, shape, strides, hex, null, null);

        /// <summary>A dump with an exception.</summary>
        /// <param name="w">The weight's prefix.</param>
        /// <param name="label">The dump's label.</param>
        /// <param name="input">Builds the input.</param>
        /// <param name="type">The Python exception type.</param>
        /// <param name="text">NumPy's text.</param>
        /// <returns>The case.</returns>
        private static Case Err(string w, string label, Func<object> input, string type, string text)
            => new(w, label, input, null, null, null, null, type, text);

        /// <summary>The "dt" inputs: <c>np.array([0.5, -0.25, 1.0, -3.0, 0.0, 2.0]).astype(dt)</c> for the floats,
        ///     <c>np.array([0, 1, 2, 3, 12, 200]).astype(dt)</c> for the integers (int64 cast: int8's 200 wraps to -56),
        ///     <c>[True, False]</c> for bool.</summary>
        /// <param name="dt">The dtype name.</param>
        /// <returns>The array.</returns>
        private static NDArray Dt(string dt) => dt switch
        {
            "bool" => np.array(new[] { true, false }),
            "float32" or "float16" => np.array(new[] { 0.5, -0.25, 1.0, -3.0, 0.0, 2.0 }).astype(np.dtype(dt)),
            _ => np.array(new long[] { 0, 1, 2, 3, 12, 200 }).astype(np.dtype(dt)),
        };

        /// <summary>The "dt … complex" input <c>np.array([0.5+1j, -2+0.25j, 0j])</c>.</summary>
        /// <returns>The array.</returns>
        private static NDArray Complex3() => np.array(new[] { new Complex(0.5, 1), new Complex(-2, 0.25), Complex.Zero });

        /// <summary>The layout probe's base, <c>np.arange(24.0).reshape(2, 3, 4) / 25</c>.</summary>
        /// <returns>The array.</returns>
        private static NDArray Base() => np.arange(24.0).reshape(2, 3, 4) / 25.0;

        /// <summary>The "perm" input: a C buffer of shape (3, 2, 4) seen through <c>transpose(1, 0, 2)</c> — shape (2, 3, 4),
        ///     element strides (4, 8, 1), neither C nor F.</summary>
        /// <param name="dt">The dtype.</param>
        /// <returns>The permuted view.</returns>
        private static NDArray Permuted(DType dt)
            => np.ascontiguousarray(np.transpose(Base(), new[] { 1, 0, 2 })).astype(dt).transpose(new[] { 1, 0, 2 });

        // The NumPy 2.4.2 dumps, made by this probe (one call per row; `dump` prints dtype | shape | element strides |
        // C-order bytes, or the exception):
        //   np:  C.chebweight(np.linspace(-1, 1, 11)[1:-1]), LG.legweight(np.linspace(-1, 1, 11)),
        //        L.lagweight(np.linspace(0, 10, 11)), H.hermweight / HE.hermeweight(np.linspace(-5, 5, 11)) — NumPy's test_weight;
        //   dt:  w(np.array(vals).astype(dt)) with the Dt values above, and w(np.array([0.5+1j, -2+0.25j, 0j]));
        //   py:  w(v) for the Python values 0.5, 2, True, -745.2, 2**70, 3037000499, 3037000500, 1.4e154, 2**2000,
        //        1e200+1e200j, complex(inf, 0), 0.5j; w(np.float16(300)), w(np.uint16(300)), w(np.array(0.5, np.float32));
        //   lay: w(np.asfortranarray(base[0])), w(<the Permuted view>) in float64 and float16,
        //        w(np.broadcast_to(base[0, 0], (3, 4))), w(np.zeros((0, 3), np.int16)).
        // A complex chebweight result is not pinned (np.sqrt's complex envelope; see ComplexChebweight_IsWithinTheSqrtEnvelope).
        private static readonly Case[] Cases =
        {
            Ok("cheb", "np cheb", () => np.linspace(-1.0, 1.0, 11)["1:-1"], "float64", new long[] { 9 }, new long[] { 1 },
               "acaaaaaaaaaafa3f000000000000f43fa3415d361a75f13fa79b07667654f03f000000000000f03fa79b07667654f03fa4415d361a75f13f000000000000f43facaaaaaaaaaafa3f"),
            Ok("leg", "np leg", () => np.linspace(-1.0, 1.0, 11), "float64", new long[] { 11 }, new long[] { 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("lag", "np lag", () => np.linspace(0.0, 10.0, 11), "float64", new long[] { 11 }, new long[] { 1 },
               "000000000000f03f38ef2c36568bd73fcc81bfa3aa52c13fafb0ceccb07da93ff43c21b855c1923f76530de03f997b3fd6d413f1514e643f8fa9249c6be14d3fad27100421fc353f5a542625f22c203f9b7c64b579cd073f"),
            Ok("herm", "np herm", () => np.linspace(-5.0, 5.0, 11), "float64", new long[] { 11 }, new long[] { 1 },
               "2ec35fa4378aae3dcb85eeba5b357e3e5a542625f22c203ff43c21b855c1923f38ef2c36568bd73f000000000000f03f38ef2c36568bd73ff43c21b855c1923f5a542625f22c203fcb85eeba5b357e3e2ec35fa4378aae3d"),
            Ok("herme", "np herme", () => np.linspace(-5.0, 5.0, 11), "float64", new long[] { 11 }, new long[] { 1 },
               "90e6683fed42cf3ead27100421fc353f17c4954650c0863fcc81bfa3aa52c13f0a966ffcb268e33f000000000000f03f0a966ffcb268e33fcc81bfa3aa52c13f17c4954650c0863fad27100421fc353f90e6683fed42cf3e"),
            Ok("cheb", "dt cheb float32", () => Dt("float32"), "float32", new long[] { 6 }, new long[] { 1 },
               "3acd933fa532843f0000807f0000c0ff0000803f0000c0ff"),
            Ok("cheb", "dt cheb float16", () => Dt("float16"), "float16", new long[] { 6 }, new long[] { 1 },
               "9f3c213c007c00fe003c00fe"),
            Ok("cheb", "dt cheb int8", () => Dt("int8"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f07f000000000000f8ff000000000000f8ff000000000000f8ff000000000000f8ff"),
            Ok("cheb", "dt cheb uint8", () => Dt("uint8"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f07f000000000000f8ff000000000000f8ff000000000000f8ff000000000000f8ff"),
            Ok("cheb", "dt cheb int16", () => Dt("int16"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f07f000000000000f8ff000000000000f8ff000000000000f8ff000000000000f8ff"),
            Ok("cheb", "dt cheb uint16", () => Dt("uint16"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f07f000000000000f8ff000000000000f8ff000000000000f8ff000000000000f8ff"),
            Ok("cheb", "dt cheb int32", () => Dt("int32"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f07f000000000000f8ff000000000000f8ff000000000000f8ff000000000000f8ff"),
            Ok("cheb", "dt cheb bool", () => Dt("bool"), "float64", new long[] { 2 }, new long[] { 1 },
               "000000000000f07f000000000000f03f"),
            Ok("leg", "dt leg float32", () => Dt("float32"), "float32", new long[] { 6 }, new long[] { 1 },
               "0000803f0000803f0000803f0000803f0000803f0000803f"),
            Ok("leg", "dt leg float16", () => Dt("float16"), "float16", new long[] { 6 }, new long[] { 1 },
               "003c003c003c003c003c003c"),
            Ok("leg", "dt leg int8", () => Dt("int8"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "dt leg uint8", () => Dt("uint8"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "dt leg int16", () => Dt("int16"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "dt leg uint16", () => Dt("uint16"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "dt leg int32", () => Dt("int32"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "dt leg bool", () => Dt("bool"), "float64", new long[] { 2 }, new long[] { 1 },
               "000000000000f03f000000000000f03f"),
            Ok("leg", "dt leg complex", () => Complex3(), "complex128", new long[] { 3 }, new long[] { 1 },
               "000000000000f03f0000000000000000000000000000f03f0000000000000000000000000000f03f0000000000000000"),
            Ok("lag", "dt lag float32", () => Dt("float32"), "float32", new long[] { 6 }, new long[] { 1 },
               "98451b3ff35aa43fb15abc3e2eafa0410000803f55950a3e"),
            Ok("lag", "dt lag float16", () => Dt("float16"), "float16", new long[] { 6 }, new long[] { 1 },
               "da38233de335054d003c5530"),
            Ok("lag", "dt lag int8", () => Dt("int8"), "float16", new long[] { 6 }, new long[] { 1 },
               "003ce33555305f2a6700007c"),
            Ok("lag", "dt lag uint8", () => Dt("uint8"), "float16", new long[] { 6 }, new long[] { 1 },
               "003c007c007c007c007c007c"),
            Ok("lag", "dt lag int16", () => Dt("int16"), "float32", new long[] { 6 }, new long[] { 1 },
               "0000803fb15abc3e55950a3e86ed4b3d622ace3600000000"),
            Ok("lag", "dt lag uint16", () => Dt("uint16"), "float32", new long[] { 6 }, new long[] { 1 },
               "0000803f0000807f0000807f0000807f0000807f0000807f"),
            Ok("lag", "dt lag int32", () => Dt("int32"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f38ef2c36568bd73fcc81bfa3aa52c13fafb0ceccb07da93f8bbc433b4cc5d93efa4c05121806e62d"),
            Err("lag", "dt lag bool", () => Dt("bool"), "TypeError", "The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead."),
            Ok("lag", "dt lag complex", () => Complex3(), "complex128", new long[] { 3 }, new long[] { 1 },
               "be4ad30033f9d43f1e784f240455e0bfae374c2f2ca31c40b220e1a1d23ffdbf000000000000f03f0000000000000080"),
            Ok("herm", "dt herm float32", () => Dt("float32"), "float32", new long[] { 6 }, new long[] { 1 },
               "7c5f473f607d703fb15abc3e916701390000803fae0a963c"),
            Ok("herm", "dt herm float16", () => Dt("float16"), "float16", new long[] { 6 }, new long[] { 1 },
               "3b3a843be3350b08003cb024"),
            Ok("herm", "dt herm int8", () => Dt("int8"), "float16", new long[] { 6 }, new long[] { 1 },
               "003ce335b0240b08007c0000"),
            Ok("herm", "dt herm uint8", () => Dt("uint8"), "float16", new long[] { 6 }, new long[] { 1 },
               "003c007c007c007c007c007c"),
            Ok("herm", "dt herm int16", () => Dt("int16"), "float32", new long[] { 6 }, new long[] { 1 },
               "0000803fb15abc3eae0a963c91670139000000000000807f"),
            Ok("herm", "dt herm uint16", () => Dt("uint16"), "float32", new long[] { 6 }, new long[] { 1 },
               "0000803f0000807f0000807f0000807f0000807f0000807f"),
            Ok("herm", "dt herm int32", () => Dt("int32"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f38ef2c36568bd73ff43c21b855c1923f5a542625f22c203f8c992393750df3320000000000000000"),
            Ok("herm", "dt herm bool", () => Dt("bool"), "float16", new long[] { 2 }, new long[] { 1 },
               "e335003c"),
            Ok("herm", "dt herm complex", () => Complex3(), "complex128", new long[] { 3 }, new long[] { 1 },
               "983fbb30164df23fa2228c169780fcbf3949cecbf692853fba4620e1c0cc903f000000000000f03f0000000000000080"),
            Ok("herme", "dt herme float32", () => Dt("float32"), "float32", new long[] { 6 }, new long[] { 1 },
               "51eb613faa1f783f98451b3f8202363c0000803f55950a3e"),
            Ok("herme", "dt herme float16", () => Dt("float16"), "float16", new long[] { 6 }, new long[] { 1 },
               "0f3bc13bda38b021003c5530"),
            Ok("herme", "dt herme int8", () => Dt("int8"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f0a966ffcb268e33fcc81bfa3aa52c13f17c4954650c0863fa70d6e6ad1aefb44686461f764840c3d"),
            Ok("herme", "dt herme uint8", () => Dt("uint8"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f0a966ffcb268e33fcc81bfa3aa52c13f17c4954650c0863fc50ef60caf757139686461f764840c3d"),
            Ok("herme", "dt herme int16", () => Dt("int16"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f0a966ffcb268e33fcc81bfa3aa52c13f17c4954650c0863fc50ef60caf757139000000000000f07f"),
            Ok("herme", "dt herme uint16", () => Dt("uint16"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f0a966ffcb268e33fcc81bfa3aa52c13f17c4954650c0863fc50ef60caf7571390000000000000000"),
            Ok("herme", "dt herme int32", () => Dt("int32"), "float64", new long[] { 6 }, new long[] { 1 },
               "000000000000f03f0a966ffcb268e33fcc81bfa3aa52c13f17c4954650c0863fc50ef60caf7571390000000000000000"),
            Ok("herme", "dt herme bool", () => Dt("bool"), "float64", new long[] { 2 }, new long[] { 1 },
               "0a966ffcb268e33f000000000000f03f"),
            Ok("herme", "dt herme complex", () => Complex3(), "complex128", new long[] { 3 }, new long[] { 1 },
               "80d0e393146ef43ff84641706952e6bf9f9aed30a65ebf3f753998e42923b13f000000000000f03f0000000000000000"),
            Ok("cheb", "py cheb 0.5", () => 0.5, "float64", new long[0], new long[0],
               "1d339045a779f23f"),
            Ok("cheb", "py cheb 2", () => 2, "float64", new long[0], new long[0],
               "000000000000f8ff"),
            Ok("cheb", "py cheb True", () => true, "float64", new long[0], new long[0],
               "000000000000f07f"),
            Ok("cheb", "py cheb -745.2", () => -745.2, "float64", new long[0], new long[0],
               "000000000000f8ff"),
            Ok("cheb", "py cheb 2**70", () => BigInteger.Pow(2, 70), "float64", new long[0], new long[0],
               "000000000000f8ff"),
            Ok("cheb", "py cheb 3037000499", () => 3037000499L, "float64", new long[0], new long[0],
               "000000000000f8ff"),
            Ok("cheb", "py cheb 3037000500", () => 3037000500L, "float64", new long[0], new long[0],
               "000000000000f8ff"),
            Ok("cheb", "py cheb 1.4e154", () => 1.4e154, "float64", new long[0], new long[0],
               "000000000000f8ff"),
            Err("cheb", "py cheb 2**2000", () => BigInteger.Pow(2, 2000), "OverflowError", "int too large to convert to float"),
            Ok("cheb", "np cheb f16 300", () => (Half)300, "float16", new long[0], new long[0],
               "00fe"),
            Ok("cheb", "np cheb u16 300", () => (char)300, "float64", new long[0], new long[0],
               "000000000000f8ff"),
            Ok("cheb", "np cheb 0d f32", () => np.array(0.5f), "float32", new long[0], new long[0],
               "3acd933f"),
            Ok("leg", "py leg 0.5", () => 0.5, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "py leg 2", () => 2, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "py leg True", () => true, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "py leg -745.2", () => -745.2, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "py leg 2**70", () => BigInteger.Pow(2, 70), "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "py leg 3037000499", () => 3037000499L, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "py leg 3037000500", () => 3037000500L, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "py leg 1.4e154", () => 1.4e154, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Err("leg", "py leg 2**2000", () => BigInteger.Pow(2, 2000), "OverflowError", "int too large to convert to float"),
            Ok("leg", "py leg 1e200+1e200j", () => new Complex(1e200, 1e200), "complex128", new long[0], new long[0],
               "000000000000f03f0000000000000000"),
            Ok("leg", "py leg inf+0j", () => new Complex(double.PositiveInfinity, 0), "complex128", new long[0], new long[0],
               "000000000000f8ff000000000000f8ff"),
            Ok("leg", "py leg 0.5j", () => new Complex(0, 0.5), "complex128", new long[0], new long[0],
               "000000000000f03f0000000000000000"),
            Ok("leg", "np leg f16 300", () => (Half)300, "float16", new long[0], new long[0],
               "003c"),
            Ok("leg", "np leg u16 300", () => (char)300, "float64", new long[0], new long[0],
               "000000000000f03f"),
            Ok("leg", "np leg 0d f32", () => np.array(0.5f), "float32", new long[0], new long[0],
               "0000803f"),
            Ok("lag", "py lag 0.5", () => 0.5, "float64", new long[0], new long[0],
               "0a966ffcb268e33f"),
            Ok("lag", "py lag 2", () => 2, "float64", new long[0], new long[0],
               "cc81bfa3aa52c13f"),
            Ok("lag", "py lag True", () => true, "float64", new long[0], new long[0],
               "38ef2c36568bd73f"),
            Ok("lag", "py lag -745.2", () => -745.2, "float64", new long[0], new long[0],
               "000000000000f07f"),
            Err("lag", "py lag 2**70", () => BigInteger.Pow(2, 70), "TypeError", "loop of ufunc does not support argument 0 of type int which has no callable exp method"),
            Ok("lag", "py lag 3037000499", () => 3037000499L, "float64", new long[0], new long[0],
               "0000000000000000"),
            Ok("lag", "py lag 3037000500", () => 3037000500L, "float64", new long[0], new long[0],
               "0000000000000000"),
            Ok("lag", "py lag 1.4e154", () => 1.4e154, "float64", new long[0], new long[0],
               "0000000000000000"),
            Err("lag", "py lag 2**2000", () => BigInteger.Pow(2, 2000), "TypeError", "loop of ufunc does not support argument 0 of type int which has no callable exp method"),
            Ok("lag", "py lag 1e200+1e200j", () => new Complex(1e200, 1e200), "complex128", new long[0], new long[0],
               "00000000000000000000000000000000"),
            Ok("lag", "py lag inf+0j", () => new Complex(double.PositiveInfinity, 0), "complex128", new long[0], new long[0],
               "00000000000000000000000000000080"),
            Ok("lag", "py lag 0.5j", () => new Complex(0, 0.5), "complex128", new long[0], new long[0],
               "507d5b062815ec3ff0054b74e8aedebf"),
            Ok("lag", "np lag f16 300", () => (Half)300, "float16", new long[0], new long[0],
               "0000"),
            Ok("lag", "np lag u16 300", () => (char)300, "float32", new long[0], new long[0],
               "0000807f"),
            Ok("lag", "np lag 0d f32", () => np.array(0.5f), "float32", new long[0], new long[0],
               "98451b3f"),
            Ok("herm", "py herm 0.5", () => 0.5, "float64", new long[0], new long[0],
               "0b82ac9eefebe83f"),
            Ok("herm", "py herm 2", () => 2, "float64", new long[0], new long[0],
               "f43c21b855c1923f"),
            Ok("herm", "py herm True", () => true, "float64", new long[0], new long[0],
               "38ef2c36568bd73f"),
            Ok("herm", "py herm -745.2", () => -745.2, "float64", new long[0], new long[0],
               "0000000000000000"),
            Err("herm", "py herm 2**70", () => BigInteger.Pow(2, 70), "TypeError", "loop of ufunc does not support argument 0 of type int which has no callable exp method"),
            Ok("herm", "py herm 3037000499", () => 3037000499L, "float64", new long[0], new long[0],
               "0000000000000000"),
            Err("herm", "py herm 3037000500", () => 3037000500L, "TypeError", "loop of ufunc does not support argument 0 of type int which has no callable exp method"),
            Err("herm", "py herm 1.4e154", () => 1.4e154, "OverflowError", "(34, 'Result too large')"),
            Err("herm", "py herm 2**2000", () => BigInteger.Pow(2, 2000), "TypeError", "loop of ufunc does not support argument 0 of type int which has no callable exp method"),
            Ok("herm", "py herm 1e200+1e200j", () => new Complex(1e200, 1e200), "complex128", new long[0], new long[0],
               "000000000000f87f000000000000f87f"),
            Ok("herm", "py herm inf+0j", () => new Complex(double.PositiveInfinity, 0), "complex128", new long[0], new long[0],
               "000000000000f87f000000000000f87f"),
            Ok("herm", "py herm 0.5j", () => new Complex(0, 0.5), "complex128", new long[0], new long[0],
               "86813e3c5e8bf43f0000000000000080"),
            Ok("herm", "np herm f16 300", () => (Half)300, "float16", new long[0], new long[0],
               "0000"),
            Ok("herm", "np herm u16 300", () => (char)300, "float32", new long[0], new long[0],
               "0000807f"),
            Ok("herm", "np herm 0d f32", () => np.array(0.5f), "float32", new long[0], new long[0],
               "7c5f473f"),
            Ok("herme", "py herme 0.5", () => 0.5, "float64", new long[0], new long[0],
               "2282ed246a3dec3f"),
            Ok("herme", "py herme 2", () => 2, "float64", new long[0], new long[0],
               "cc81bfa3aa52c13f"),
            Ok("herme", "py herme True", () => true, "float64", new long[0], new long[0],
               "0a966ffcb268e33f"),
            Ok("herme", "py herme -745.2", () => -745.2, "float64", new long[0], new long[0],
               "0000000000000000"),
            Ok("herme", "py herme 2**70", () => BigInteger.Pow(2, 70), "float64", new long[0], new long[0],
               "0000000000000000"),
            Ok("herme", "py herme 3037000499", () => 3037000499L, "float64", new long[0], new long[0],
               "0000000000000000"),
            Ok("herme", "py herme 3037000500", () => 3037000500L, "float64", new long[0], new long[0],
               "0000000000000000"),
            Err("herme", "py herme 1.4e154", () => 1.4e154, "OverflowError", "(34, 'Result too large')"),
            Err("herme", "py herme 2**2000", () => BigInteger.Pow(2, 2000), "OverflowError", "int too large to convert to float"),
            Ok("herme", "py herme 1e200+1e200j", () => new Complex(1e200, 1e200), "complex128", new long[0], new long[0],
               "000000000000f8ff000000000000f8ff"),
            Ok("herme", "py herme inf+0j", () => new Complex(double.PositiveInfinity, 0), "complex128", new long[0], new long[0],
               "000000000000f8ff000000000000f8ff"),
            Ok("herme", "py herme 0.5j", () => new Complex(0, 0.5), "complex128", new long[0], new long[0],
               "cdf5b6456021f23f0000000000000080"),
            Ok("herme", "np herme f16 300", () => (Half)300, "float16", new long[0], new long[0],
               "0000"),
            Ok("herme", "np herme u16 300", () => (char)300, "float64", new long[0], new long[0],
               "0000000000000000"),
            Ok("herme", "np herme 0d f32", () => np.array(0.5f), "float32", new long[0], new long[0],
               "51eb613f"),
            Ok("cheb", "lay cheb F", () => np.asfortranarray(Base()[0]), "float64", new long[] { 3, 4 }, new long[] { 1, 3 },
               "000000000000f03fb42b68de4703f03fddecbda22b0df03f8a188146d01df03f37a82d187535f03fa79b07667654f03f45bad888517bf03fabaaaaaaaaaaf03f8d99e2d554e3f03f15bca9205d26f13fa3415d361a75f13f56d0ee2e42d1f13f"),
            Ok("cheb", "lay cheb perm", () => Permuted(np.float64), "float64", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "000000000000f03fb42b68de4703f03fddecbda22b0df03f8a188146d01df03f37a82d187535f03fa79b07667654f03f45bad888517bf03fabaaaaaaaaaaf03f8d99e2d554e3f03f15bca9205d26f13fa3415d361a75f13f56d0ee2e42d1f13f6dec58c1093df23fe3d144c151bbf23f3e269128eb4ff33f000000000000f43fb1a59027bbd2f43f0b92f4c360d2f53fe1b286e53c0ef73f316c62ca489ef83facaaaaaaaaaafa3fb7b52c35077dfd3fade2b05dd1d70040928289ff93690440"),
            Ok("cheb", "lay cheb perm f16", () => Permuted(np.float16), "float16", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "003c013c043c073c0d3c153c1f3c2b3c393c4a3c5d3c743c8f3caf3cd53c003d343d753dc53d283ea93e5e3f35401a41"),
            Ok("cheb", "lay cheb bcast", () => np.broadcast_to(Base()[0, 0], new Shape(3, 4)), "float64", new long[] { 3, 4 }, new long[] { 4, 1 },
               "000000000000f03fb42b68de4703f03fddecbda22b0df03f8a188146d01df03f000000000000f03fb42b68de4703f03fddecbda22b0df03f8a188146d01df03f000000000000f03fb42b68de4703f03fddecbda22b0df03f8a188146d01df03f"),
            Ok("cheb", "lay cheb empty", () => np.zeros(new Shape(0, 3), np.int16), "float64", new long[] { 0, 3 }, new long[0],
               ""),
            Ok("leg", "lay leg F", () => np.asfortranarray(Base()[0]), "float64", new long[] { 3, 4 }, new long[] { 1, 3 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "lay leg perm", () => Permuted(np.float64), "float64", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "lay leg perm f16", () => Permuted(np.float16), "float16", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c003c"),
            Ok("leg", "lay leg bcast", () => np.broadcast_to(Base()[0, 0], new Shape(3, 4)), "float64", new long[] { 3, 4 }, new long[] { 4, 1 },
               "000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f000000000000f03f"),
            Ok("leg", "lay leg empty", () => np.zeros(new Shape(0, 3), np.int16), "float64", new long[] { 0, 3 }, new long[0],
               ""),
            Ok("lag", "lay lag F", () => np.asfortranarray(Base()[0]), "float64", new long[] { 3, 4 }, new long[] { 1, 3 },
               "000000000000f03f8d0b707ec9beee3f8546c44a2b8aed3fd855bbf7a661ec3f502e230dc344eb3f596116d60a33ea3fa9e72c310e2ce93fd9d38c62612fe83fce1bc8e79c3ce73f7064754d5d53e63fee7072064373e53fef84bf44f29be43f"),
            Ok("lag", "lay lag perm", () => Permuted(np.float64), "float64", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "000000000000f03f8d0b707ec9beee3f8546c44a2b8aed3fd855bbf7a661ec3f502e230dc344eb3f596116d60a33ea3fa9e72c310e2ce93fd9d38c62612fe83fce1bc8e79c3ce73f7064754d5d53e63fee7072064373e53fef84bf44f29be43fc7b0e3d312cde33f9d9fcaf44f06e33f191a0d3b5847e23fe404966bdd8fe13fce30965c94dfe03fdddbb9d63436e03f138c26eff226df3f7a765c2a3feedd3f88db8145cec1dc3f55e2dd2c25a1db3fcd2b27a0cd8bda3f25a412025681d93f"),
            Ok("lag", "lay lag perm f16", () => Permuted(np.float16), "float16", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "003cb03b633b183bd13a8d3a4b3a0c3acf3995395d392739f338c2389238643838380d38c9377c373137e936a3366036"),
            Ok("lag", "lay lag bcast", () => np.broadcast_to(Base()[0, 0], new Shape(3, 4)), "float64", new long[] { 3, 4 }, new long[] { 4, 1 },
               "000000000000f03f8d0b707ec9beee3f8546c44a2b8aed3fd855bbf7a661ec3f000000000000f03f8d0b707ec9beee3f8546c44a2b8aed3fd855bbf7a661ec3f000000000000f03f8d0b707ec9beee3f8546c44a2b8aed3fd855bbf7a661ec3f"),
            Ok("lag", "lay lag empty", () => np.zeros(new Shape(0, 3), np.int16), "float32", new long[] { 0, 3 }, new long[0],
               ""),
            Ok("herm", "lay herm F", () => np.asfortranarray(Base()[0]), "float64", new long[] { 3, 4 }, new long[] { 1, 3 },
               "000000000000f03f8d845e3de7f2ef3f9a57dc15bdcbef3f83ce6b67e18aef3fad3fdf46f230ef3f8d0b707ec9beee3fa008b51f7935ee3f53985e3c4796ed3f00a3a0dda8e2ec3f342519563c1cec3f502e230dc344eb3f4c30cfe41a5eea3f"),
            Ok("herm", "lay herm perm", () => Permuted(np.float64), "float64", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "000000000000f03f8d845e3de7f2ef3f9a57dc15bdcbef3f83ce6b67e18aef3fad3fdf46f230ef3f8d0b707ec9beee3fa008b51f7935ee3f53985e3c4796ed3f00a3a0dda8e2ec3f342519563c1cec3f502e230dc344eb3f4c30cfe41a5eea3f897d1e5d376ae93f8e729c971a6be83f7f1d075ece62e73f7064754d5d53e63f3a645a45cc3ee53f896a03371427e43f33add26e1c0ee33f2379a36cb5f5e13fce30965c94dfe03f7ed23a7a9e9adf3f7b145875b380dd3fbc23ef890574db3f"),
            Ok("herm", "lay herm perm f16", () => Permuted(np.float16), "float16", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "003cfd3bf33be33bcc3bb03b8d3b663b393b073bd13a983a5b3a1b3ad939953950390939c3387e383838e7376037dd36"),
            Ok("herm", "lay herm bcast", () => np.broadcast_to(Base()[0, 0], new Shape(3, 4)), "float64", new long[] { 3, 4 }, new long[] { 4, 1 },
               "000000000000f03f8d845e3de7f2ef3f9a57dc15bdcbef3f83ce6b67e18aef3f000000000000f03f8d845e3de7f2ef3f9a57dc15bdcbef3f83ce6b67e18aef3f000000000000f03f8d845e3de7f2ef3f9a57dc15bdcbef3f83ce6b67e18aef3f"),
            Ok("herm", "lay herm empty", () => np.zeros(new Shape(0, 3), np.int16), "float32", new long[] { 0, 3 }, new long[0],
               ""),
            Ok("herme", "lay herme F", () => np.asfortranarray(Base()[0]), "float64", new long[] { 3, 4 }, new long[] { 1, 3 },
               "000000000000f03f34f605f372f9ef3fdf54eed6d3e5ef3f9e3dcabb3ac5ef3f9c8fbd85cf97ef3f5becad9bc95def3f9e99b0766f17ef3f214d771316c5ee3fac794f472067ee3fd98395fafdfded3f8546c44a2b8aed3ff48f88962f0ced3f"),
            Ok("herme", "lay herme perm", () => Permuted(np.float64), "float64", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "000000000000f03f34f605f372f9ef3fdf54eed6d3e5ef3f9e3dcabb3ac5ef3f9c8fbd85cf97ef3f5becad9bc95def3f9e99b0766f17ef3f214d771316c5ee3fac794f472067ee3fd98395fafdfded3f8546c44a2b8aed3ff48f88962f0ced3fc0ef76769c84ec3f759f2fa50cf4eb3f40ffdfd9225beb3f90b32a9888baea3f54b78ef8ec12ea3f27ab6a6c0365e93fe270b48082b1e83f98086ea222f9e73fcd1bc8e79c3ce73f6f42c1e0a97ce63feb89e77100bae53f2af2b0bb54f5e43f"),
            Ok("herme", "lay herme perm f16", () => Permuted(np.float16), "float16", new long[] { 2, 3, 4 }, new long[] { 4, 8, 1 },
               "003cfe3bf93bf13be63bd73bc63bb13b9a3b7f3b633b433b213bfd3ad73aaf3a853a593a2c3aff39cf399f396e393d39"),
            Ok("herme", "lay herme bcast", () => np.broadcast_to(Base()[0, 0], new Shape(3, 4)), "float64", new long[] { 3, 4 }, new long[] { 4, 1 },
               "000000000000f03f34f605f372f9ef3fdf54eed6d3e5ef3f9e3dcabb3ac5ef3f000000000000f03f34f605f372f9ef3fdf54eed6d3e5ef3f9e3dcabb3ac5ef3f000000000000f03f34f605f372f9ef3fdf54eed6d3e5ef3f9e3dcabb3ac5ef3f"),
            Ok("herme", "lay herme empty", () => np.zeros(new Shape(0, 3), np.int16), "float64", new long[] { 0, 3 }, new long[0],
               ""),
        };

        /// <summary>
        ///     Every NumPy dump: NumPy's own <c>test_weight</c> inputs, the dtype rules of each statement (<c>x**2</c> is
        ///     <c>np.square</c> in x's own dtype: a bool squares to int8, uint8's 200 wraps; <c>-x</c> wraps an unsigned x),
        ///     Python- and NumPy-scalar arguments (CPython's arithmetic until the first ufunc, the int64 / uint64 / object
        ///     boundaries of np.exp's conversion), and the result layouts (an F-ordered or permuted x keeps its memory order,
        ///     a broadcast x gives C, an empty x its dtype).
        /// </summary>
        [TestMethod]
        public void NumPyDumps_AreReproduced()
        {
            foreach (var c in Cases)
            {
                if (c.ErrorType != null)
                {
                    var e = AssertThrowsAny(() => Weights[c.Weight](c.Input()), c.Label);
                    Assert.AreEqual(c.ErrorType, PythonName(e), $"{c.Label}: exception type");
                    Assert.AreEqual(c.ErrorText, e.Message, $"{c.Label}: exception text");
                    continue;
                }
                AssertCase(c, Weights[c.Weight](c.Input()));
            }
        }

        /// <summary>
        ///     A Python list or tuple, None or a str is never converted: the first operator of NumPy's statement meets it,
        ///     and Python raises its own TypeError before NumPy sees the value (probed, NumPy 2.4.2 / CPython 3.12). The C#
        ///     spellings of a Python list — an <c>object[]</c>, a <see cref="List{T}"/>, a jagged array — all read as one.
        /// </summary>
        [TestMethod]
        public void PythonObjects_RaisePythonsOperatorTypeErrors()
        {
            var lists = new object[] { new object[] { 0.5 }, new List<double> { 0.5 }, new[] { new[] { 0.5 } }, new object[0] };
            var expected = new Dictionary<string, Func<string, string>>
            {
                ["cheb"] = t => $"unsupported operand type(s) for +: 'float' and '{t}'",
                ["leg"] = t => t == "NoneType" ? "unsupported operand type(s) for *: 'NoneType' and 'float'"
                                              : "can't multiply sequence by non-int of type 'float'",
                ["lag"] = t => $"bad operand type for unary -: '{t}'",
                ["herm"] = t => $"unsupported operand type(s) for ** or pow(): '{t}' and 'int'",
                ["herme"] = t => $"unsupported operand type(s) for ** or pow(): '{t}' and 'int'",
            };
            foreach (var (w, text) in expected)
            {
                foreach (var list in lists)
                    AssertTypeError(() => Weights[w](list), text("list"), $"{w}({list.GetType().Name})");
                AssertTypeError(() => Weights[w](ValueTuple.Create(0.5)), text("tuple"), $"{w}(tuple)");
                AssertTypeError(() => Weights[w](null), text("NoneType"), $"{w}(None)");
                AssertTypeError(() => Weights[w]("ab"), text("str"), $"{w}(str)");
            }
        }

        /// <summary>
        ///     A string array (U, S or StringDType) reaches the first statement's ufunc, which has no loop for it: NumPy's
        ///     texts, the Python float named <c>dtype('float64')</c> beside U / S and the abstract <c>_PyFloatDType</c>
        ///     beside StringDType (probed). <c>np.square</c> of U / S is the legacy resolver's "not supported" text.
        /// </summary>
        [TestMethod]
        public void StringArrays_RaiseTheFirstUfuncsNoLoopError()
        {
            var u = np.array(new[] { "ab" });
            var s = u.astype(np.dtype("S2"));
            var t = u.astype(np.dtype("T"));
            const string Square = "ufunc 'square' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''";
            var cases = new (string W, NDArray X, string Text)[]
            {
                ("cheb", u, "ufunc 'add' did not contain a loop with signature matching types (dtype('float64'), dtype('<U2')) -> None"),
                ("cheb", s, "ufunc 'add' did not contain a loop with signature matching types (dtype('float64'), dtype('S2')) -> None"),
                ("cheb", t, "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes._PyFloatDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
                ("leg", u, "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U2'), dtype('float64')) -> None"),
                ("leg", s, "ufunc 'multiply' did not contain a loop with signature matching types (dtype('S2'), dtype('float64')) -> None"),
                ("leg", t, "ufunc 'multiply' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes._PyFloatDType'>) -> None"),
                ("lag", u, "ufunc 'negative' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> None"),
                ("lag", s, "ufunc 'negative' did not contain a loop with signature matching types <class 'numpy.dtypes.BytesDType'> -> None"),
                ("lag", t, "ufunc 'negative' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> None"),
                ("herm", u, Square),
                ("herm", s, Square),
                ("herm", t, "ufunc 'square' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> None"),
                ("herme", u, Square),
                ("herme", s, Square),
                ("herme", t, "ufunc 'square' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> None"),
                ("cheb", u[0], "ufunc 'add' did not contain a loop with signature matching types (dtype('float64'), dtype('<U2')) -> None"),
            };
            foreach (var (w, x, text) in cases)
                AssertTypeError(() => Weights[w](x), text, $"{w}({x.dtype.name}, ndim {x.ndim})");
        }

        /// <summary><c>lagweight</c> of a bool array is NumPy's boolean-negative TypeError, from either route (an empty
        ///     bool array too: the loop is refused before any element is read).</summary>
        [TestMethod]
        public void BoolArray_Lagweight_RaisesTheBooleanNegativeError()
        {
            const string Text = "The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.";
            AssertTypeError(() => np.polynomial.laguerre.lagweight(np.array(new[] { true, false })), Text, "1-d");
            AssertTypeError(() => np.polynomial.laguerre.lagweight(np.zeros(new Shape(0), np.bool_)), Text, "empty");
            AssertTypeError(() => np.polynomial.laguerre.lagweight(np.array(true)), Text, "0-d");
        }

        /// <summary>
        ///     Complex <c>chebweight</c> inherits <c>np.sqrt</c>'s complex accuracy: NumSharp's complex square root is within a
        ///     few ULP of MSVC's <c>csqrt</c>, not bit-identical, so these values are held to 4 ULP of NumPy's (probed:
        ///     <c>chebweight(np.array([0.5+1j, -2+0.25j, 0j]))</c>). The special values are exact.
        /// </summary>
        [TestMethod]
        public void ComplexChebweight_IsWithinTheSqrtEnvelope()
        {
            var r = np.polynomial.chebyshev.chebweight(Complex3());
            Assert.AreEqual("complex128", r.dtype.name);
            var expected = new (ulong Re, ulong Im)[]
            {
                (0x3fe5c8e7dba109e2, 0x3fc7241c7d3e6aba),
                (0x3fb7bc4980f9102b, 0xbfe1ec034f1fcfc4),
                (0x3ff0000000000000, 0x0000000000000000),
            };
            for (int i = 0; i < expected.Length; i++)
            {
                var z = r.GetAtIndex<Complex>(i);
                Assert.IsTrue(UlpDistance(BitConverter.DoubleToInt64Bits(z.Real), (long)expected[i].Re) <= 4, $"[{i}] real part");
                Assert.IsTrue(UlpDistance(BitConverter.DoubleToInt64Bits(z.Imaginary), (long)expected[i].Im) <= 4, $"[{i}] imaginary part");
            }
            var special = np.polynomial.chebyshev.chebweight(new Complex(double.PositiveInfinity, 0));
            Assert.IsTrue(double.IsNaN(special.GetAtIndex<Complex>(0).Real) && double.IsNaN(special.GetAtIndex<Complex>(0).Imaginary));
        }

        /// <summary>
        ///     Decimal has no NumPy dtype: a decimal x runs NumPy's statements on the house decimal routines (the definition
        ///     of a NumSharp-only weight), so where a float result would be NaN or infinite, decimal — which has neither —
        ///     raises.
        /// </summary>
        [TestMethod]
        public void Decimal_RunsTheStatementsOnTheHouseDecimalRoutines()
        {
            var x = np.array(new[] { 0.5m, -0.25m });
            var leg = np.polynomial.legendre.legweight(x);
            Assert.AreEqual(NPTypeCode.Decimal, leg.typecode);
            Assert.AreEqual(1m, leg.GetAtIndex<decimal>(0));
            Assert.AreEqual(1m, leg.GetAtIndex<decimal>(1));
            Assert.AreEqual(NPTypeCode.Decimal, np.polynomial.chebyshev.chebweight(x).typecode);
            Assert.AreEqual(NPTypeCode.Decimal, np.polynomial.hermite.hermweight(0.5m).typecode);   // a NumPy scalar: 0-d result
            Assert.AreEqual(0, np.polynomial.hermite.hermweight(0.5m).ndim);
            Assert.ThrowsException<OverflowException>(() => np.polynomial.chebyshev.chebweight(np.array(new[] { 2m })));
            Assert.ThrowsException<DivideByZeroException>(() => np.polynomial.chebyshev.chebweight(np.array(new[] { 1m })));
            Assert.ThrowsException<OverflowException>(() => np.polynomial.laguerre.lagweight(np.array(new[] { -1000m })));
        }

        /// <summary>NumSharp's char dtype is NumPy's uint16 (the house proxy): every weight gives the uint16 result.</summary>
        [TestMethod]
        public void Char_ComputesAsUint16()
        {
            var values = new long[] { 0, 1, 2, 3, 12, 200, 255, 256, 300, 65535 };
            NDArray u16 = np.array(values).astype(np.uint16), ch = np.array(values).astype(np.dtype("char"));
            foreach (var (w, f) in Weights)
            {
                NDArray a = f(u16), b = f(ch);
                Assert.AreEqual(a.dtype.name, b.dtype.name, w);
                Assert.AreEqual(Hex(a), Hex(b), w);
            }
        }

        /// <summary>
        ///     A <c>numpy.matrix</c> x is computed as its plain array (elementwise): NumPy's own answer would run the matrix
        ///     operators — <c>x**2</c> the matrix power, <c>np.sqrt(1.0 + x)*np.sqrt(1.0 - x)</c> a matrix product — which no
        ///     weight's definition intends, and <c>mapdomain</c> makes the same choice. The result is a plain array.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Matrix_IsComputedAsItsPlainArray()
        {
            var plain = np.array(new[,] { { 0.5, 1.5 }, { -0.25, 2.0 } });
            var m = np.matrix(plain);
            foreach (var (w, f) in Weights)
            {
                var r = f(m);
                Assert.IsNotInstanceOfType(r, typeof(NDMatrix), w);
                Assert.AreEqual(Hex(f(plain)), Hex(r), w);
            }
        }

        /// <summary>
        ///     The two routes compute the same bytes: for every dtype the fused <c>np.evaluate</c> pass serves, its result
        ///     equals the statements replayed on the house ufuncs (NDExpr types every node as NumPy does; a divergence here is
        ///     a route bug the NumPy dumps might not reach), errors included. Values cover the vector body and tail, signed
        ///     zeros, infinities, NaN and the integer extremes.
        /// </summary>
        [TestMethod]
        public void FusedTree_AgreesWithTheStatements_ForEveryFusedDtype()
        {
            double inf = double.PositiveInfinity, nan = BitConverter.Int64BitsToDouble(0x7ff8000000000000L);
            var floats = new[] { 0.5, -0.25, 0.0, -0.0, 1.0, -1.0, 2.0, 3.5, -7.0, inf, -inf, nan, 1e-3, 0.9999, 1e30, -1e30, 300.0, 65504.0, 26.6 };
            var ints = new long[] { 0, 1, -1, 2, -2, 3, 100, -100, 127, -128, 200, 255, 1000, 30000, -30000, 65535, int.MaxValue, int.MinValue, 3037000499 };
            var inputs = new List<NDArray>();
            foreach (var dt in new[] { "float64", "float32" })
                inputs.Add(np.array(floats).astype(np.dtype(dt)));
            foreach (var dt in new[] { "int8", "uint8", "int16", "uint16", "char", "int32", "uint32", "int64", "uint64" })
                inputs.Add(np.array(ints).astype(np.dtype(dt)));
            inputs.Add(np.array(new[] { true, false, true, true, false }));
            inputs.Add(np.array(new[] { new Complex(0.5, 1), new Complex(-inf, 1), new Complex(inf, 0), new Complex(nan, 0), Complex.Zero,
                new Complex(-0.0, 0.0), new Complex(3, -4), new Complex(0.25, -0.5), new Complex(1, inf) }));
            foreach (var x in inputs)
                foreach (var (w, basis) in Bases)
                {
                    string label = $"{w}({x.dtype.name})";
                    Exception fusedError = null, statementError = null;
                    NDArray fused = null, statements = null;
                    try { fused = np.evaluate(NDPolyWeight.Tree(basis, x)); } catch (Exception e) { fusedError = e; }
                    try { statements = NDPolyWeight.Statements(basis, PolyNumber.FromArray(x)).Array; } catch (Exception e) { statementError = e; }
                    if (fusedError != null || statementError != null)
                    {
                        Assert.IsNotNull(fusedError, $"{label}: only the statements raised: {statementError?.Message}");
                        Assert.IsNotNull(statementError, $"{label}: only the fused pass raised: {fusedError.Message}");
                        Assert.AreEqual(statementError.GetType(), fusedError.GetType(), label);
                        Assert.AreEqual(statementError.Message, fusedError.Message, label);
                        continue;
                    }
                    Assert.AreEqual(statements.dtype.name, fused.dtype.name, label);
                    Assert.AreEqual(Hex(statements), Hex(fused), label);
                }
        }

        /// <summary>
        ///     CPython 3.12's <c>x ** 2</c> on a Python value (<see cref="PyScalar.PowTwo"/>): a float calls the platform
        ///     <c>pow</c> after <c>float_pow</c>'s special cases — NOT the product (ucrtbase's <c>pow</c> rounds
        ///     <c>2.3315311434106746e+132**2</c> one ULP below <c>x*x</c>), a NaN comes back as it is (a signalling NaN keeps its
        ///     bits), an infinite square raises; a complex runs <c>c_powu</c>'s two naive products, so <c>(inf+0j)**2</c> is
        ///     nan+nanj, and an infinite part of the result raises; an int squares exactly, a bool to a plain int.
        /// </summary>
        [TestMethod]
        public void PowTwo_PythonValues_AreCPythonsPow()
        {
            static double D(long bits) => BitConverter.Int64BitsToDouble(bits);
            static long Bits(in PyScalar p) => BitConverter.DoubleToInt64Bits(p.F);

            Assert.AreEqual(0x76e594495e2c9536L, Bits(PyScalar.PowTwo(PyScalar.Float(2.3315311434106746e+132))), "pow, not the product");
            Assert.AreEqual(0x7ff4000000000001L, Bits(PyScalar.PowTwo(PyScalar.Float(D(0x7ff4000000000001L)))), "a signalling NaN is returned as it is");
            Assert.AreEqual(0L, Bits(PyScalar.PowTwo(PyScalar.Float(-0.0))), "(-0.0)**2 is +0.0");
            Assert.AreEqual(double.PositiveInfinity, PyScalar.PowTwo(PyScalar.Float(double.NegativeInfinity)).F);
            Assert.AreEqual(1.0, PyScalar.PowTwo(PyScalar.Float(-1.0)).F);
            Assert.AreEqual(0.0, PyScalar.PowTwo(PyScalar.Float(1e-200)).F, "an underflow to zero is no error");
            Assert.IsTrue(PyScalar.PowTwo(PyScalar.Float(1e-160)).F > 0, "a subnormal square is no error on ucrtbase");
            var overflow = Assert.ThrowsException<OverflowException>(() => PyScalar.PowTwo(PyScalar.Float(1.4e154)));
            Assert.AreEqual("(34, 'Result too large')", overflow.Message);

            var inf2 = PyScalar.PowTwo(PyScalar.Cplx(new Complex(double.PositiveInfinity, 0)));
            Assert.AreEqual(PyKind.Complex, inf2.Kind);
            Assert.AreEqual(unchecked((long)0xfff8000000000000UL), BitConverter.DoubleToInt64Bits(inf2.C.Real), "(inf+0j)**2: default NaN");
            Assert.AreEqual(unchecked((long)0xfff8000000000000UL), BitConverter.DoubleToInt64Bits(inf2.C.Imaginary));
            var c = Assert.ThrowsException<OverflowException>(() => PyScalar.PowTwo(PyScalar.Cplx(new Complex(0, 1.4e154))));
            Assert.AreEqual("complex exponentiation", c.Message);

            Assert.AreEqual(BigInteger.Pow(2, 140), PyScalar.PowTwo(PyScalar.Int(BigInteger.Pow(2, 70))).I);
            var b = PyScalar.PowTwo(PyScalar.Bool(true));
            Assert.AreEqual(PyKind.Int, b.Kind);
            Assert.IsFalse(b.IsBool, "True**2 is the int 1");
            Assert.AreEqual(BigInteger.One, b.I);
        }

        /// <summary>
        ///     <see cref="PolyNumber.PowTwo"/> on NumPy scalars is scalarmath's power (probed, NumPy 2.4.2): float64 is
        ///     <c>pow</c> and float32 <c>powf</c> — not the product (<c>np.float32(2.9465066e-23)**2</c> is 0, its square 1e-45),
        ///     float16 <c>powf</c> rounded to float16 (300**2 is inf), uint16 / char wrap (300**2 is 24464), int8 wraps (12**2 is
        ///     -112), a bool runs the int64 loop (<c>np.True_**2</c> is <c>np.int64(1)</c>), a complex zero base is (+0, +0)
        ///     whatever its signs, and (1+2j)**2 is (-3+4j).
        /// </summary>
        [TestMethod]
        public unsafe void PowTwo_NumPyScalars_AreScalarmathsPower()
        {
            static string H(in PolyNumber p) => Hex(p.ToNDArray()) + " " + p.ToNDArray().dtype.name;

            Assert.AreEqual("36952c5e4994e576 float64", H(PolyNumber.PowTwo(PolyNumber.FromScalar(2.3315311434106746e+132))));
            Assert.AreEqual("00000000 float32", H(PolyNumber.PowTwo(PolyNumber.FromScalar(2.9465066e-23f))));
            Assert.AreEqual("007c float16", H(PolyNumber.PowTwo(PolyNumber.FromScalar((Half)300))));
            Assert.AreEqual("905f char", H(PolyNumber.PowTwo(PolyNumber.FromScalar((char)300))));
            Assert.AreEqual("90 int8", H(PolyNumber.PowTwo(PolyNumber.FromScalar((sbyte)12))));
            Assert.AreEqual("0100000000000000 int64", H(PolyNumber.PowTwo(PolyNumber.FromScalar(true))));
            Assert.AreEqual("00000000000000000000000000000000 complex128", H(PolyNumber.PowTwo(PolyNumber.FromScalar(new Complex(-0.0, 0.0)))));
            Assert.AreEqual("00000000000000000000000000000000 complex128", H(PolyNumber.PowTwo(PolyNumber.FromScalar(new Complex(-0.0, -0.0)))));
            Assert.AreEqual("00000000000008c00000000000001040 complex128", H(PolyNumber.PowTwo(PolyNumber.FromScalar(new Complex(1, 2)))));
        }

        /// <summary>
        ///     <see cref="PolyNumber.FloatLoop"/> restates the engine's float-producing unary dtype rule over a dtype code
        ///     (<c>DefaultEngine.ResolveUnaryFloatReturnType</c>); the scalar route trusts it to pick np.sqrt's / np.exp's
        ///     loop, so the two must agree for every dtype.
        /// </summary>
        [TestMethod]
        public void FloatLoop_EqualsTheEnginesUnaryFloatRule()
        {
            var engine = (NumSharp.Backends.DefaultEngine)np.zeros(new Shape(1)).TensorEngine;
            foreach (NPTypeCode t in new[] { NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                         NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half,
                         NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex })
                Assert.AreEqual(engine.ResolveUnaryFloatReturnType(np.zeros(new Shape(1), t), null), PolyNumber.FloatLoop(t), t.ToString());
        }

        /// <summary>
        ///     A scalar-sized operand runs the house's contiguous unary kernel on one stack element (no 0-d arrays); its result
        ///     must be the bits <c>np.sqrt</c> / <c>np.exp</c> give the same value as a 0-d array, for every dtype and the
        ///     special values (NaN, ±inf, ±0, a subnormal, the integer extremes).
        /// </summary>
        [TestMethod]
        public void ScalarKernelRoute_EqualsTheUfuncOnAZeroDimArray()
        {
            double inf = double.PositiveInfinity, nan = BitConverter.Int64BitsToDouble(0x7ff8000000000000L);
            var doubles = new[] { 0.5, -0.25, 0.0, -0.0, 2.0, -745.2, 709.8, inf, -inf, nan, 5e-324, 1e30 };
            var ints = new long[] { 0, 1, -1, 2, 127, -128, 255, 32767, -32768, 65535, int.MaxValue, int.MinValue, long.MaxValue };
            var operands = new List<NDArray>();
            foreach (var dt in new[] { "float64", "float32", "float16" })
                foreach (var v in doubles)
                    operands.Add(np.array(new[] { v }).astype(np.dtype(dt))[0]);
            foreach (var dt in new[] { "int8", "uint8", "int16", "uint16", "char", "int32", "uint32", "int64", "uint64" })
                foreach (var v in ints)
                    operands.Add(np.array(new[] { v }).astype(np.dtype(dt))[0]);
            operands.Add(np.array(true));
            operands.Add(np.array(false));
            foreach (var v in new[] { new Complex(0.5, 1), new Complex(-2, 0.25), new Complex(inf, 0), new Complex(nan, 0), Complex.Zero, new Complex(-0.0, 0.0) })
                operands.Add(np.array(v));
            operands.Add(np.array(0.25m));
            foreach (var x in operands)
                foreach (var (op, name, f) in new (UnaryOp, string, Func<NDArray, NDArray>)[]
                         { (UnaryOp.Sqrt, "sqrt", a => np.sqrt(a)), (UnaryOp.Exp, "exp", a => np.exp(a)) })
                {
                    string label = $"{name}({x.dtype.name} {x})";
                    NDArray viaUfunc;
                    try { viaUfunc = f(x); }
                    catch (Exception e)
                    {
                        // A value the decimal routines cannot represent raises in both routes alike.
                        var other = AssertThrowsAny(() => PolyNumber.FloatUfunc(op, name, f, PolyNumber.FromArray(x)), label);
                        Assert.AreEqual(e.GetType(), other.GetType(), label);
                        continue;
                    }
                    var viaKernel = PolyNumber.FloatUfunc(op, name, f, PolyNumber.FromArray(x)).ToNDArray();
                    Assert.AreEqual(viaUfunc.dtype.name, viaKernel.dtype.name, label);
                    Assert.AreEqual(Hex(viaUfunc), Hex(viaKernel), label);
                }
        }

        /// <summary>
        ///     The two shortcut routes of an ndarray x give the bytes of the plain fused pass: a broadcast x is computed once on
        ///     its un-broadcast core and copied (the weight is elementwise and deterministic), and an integer or bool x is cast
        ///     once into the first float loop (NumPy's own buffered cast) — for every weight, every integer dtype and a row,
        ///     column and full broadcast. The layout of a broadcast result is NumPy's (C here: the broadcast axes abstain).
        /// </summary>
        [TestMethod]
        public void ShortcutRoutes_EqualThePlainFusedPass()
        {
            var ints = np.array(new long[] { 0, 1, -1, 2, -2, 3, 100, -100, 127, -128, 200, 255, 1000, 30000, -30000, 65535, int.MaxValue, int.MinValue });
            foreach (var (w, basis) in Bases)
            {
                foreach (var dt in new[] { "int8", "uint8", "int16", "uint16", "char", "int32", "uint32", "int64", "uint64", "bool" })
                {
                    var x = dt == "bool" ? np.array(new[] { true, false, true }) : ints.astype(np.dtype(dt));
                    string label = $"{w}({dt})";
                    if (w == "lag" && dt == "bool")
                    {
                        AssertTypeError(() => Weights[w](x), "The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.", label);
                        continue;
                    }
                    var expected = np.evaluate(NDPolyWeight.Tree(basis, x));
                    var actual = Weights[w](x);
                    Assert.AreEqual(expected.dtype.name, actual.dtype.name, label);
                    Assert.AreEqual(Hex(expected), Hex(actual), label);
                }
                var row = np.array(new[] { 0.5, -0.25, 3.0, double.NaN, 0.0 });
                var col = np.array(new[] { 0.75, -1.5, 2.0 }).reshape(3, 1);
                foreach (var (label, b, dense) in new (string, NDArray, NDArray)[]
                         {
                             ("row", np.broadcast_to(row, new Shape(4, 5)), np.ascontiguousarray(np.broadcast_to(row, new Shape(4, 5)))),
                             ("column", np.broadcast_to(col, new Shape(3, 6)), np.ascontiguousarray(np.broadcast_to(col, new Shape(3, 6)))),
                             ("full", np.broadcast_to(np.array(0.5), new Shape(7)), np.full(new Shape(7), 0.5)),
                         })
                {
                    var actual = Weights[w](b);
                    var expected = Weights[w](dense);
                    Assert.AreEqual(expected.dtype.name, actual.dtype.name, $"{w} {label}");
                    Assert.AreEqual(Hex(expected), Hex(actual), $"{w} {label}");
                    Assert.IsTrue(actual.Shape.IsContiguous, $"{w} {label}: C-ordered like NumPy's");
                    Assert.IsTrue(actual.flags.writeable && actual.flags.owndata, $"{w} {label}: a fresh array");
                }
            }
        }

        /// <summary>
        ///     A broadcast x's result is computed on the un-broadcast core and copied into NumPy's ufunc layout, which keeps
        ///     the memory order of the axes that are not broadcast (probed: <c>hermweight(np.broadcast_to(F(3, 4), (2, 3, 4)))</c>
        ///     has element strides (12, 1, 3) — the broadcast axis outermost, the others F-ordered — and a broadcast middle axis
        ///     over an F (3, 4) gives (1, 12, 3)); the values are those of the dense computation.
        /// </summary>
        [TestMethod]
        public void BroadcastResult_KeepsNumPysLayout()
        {
            var f = np.asfortranarray(np.arange(12.0).reshape(3, 4) / 13.0);
            var cases = new (NDArray X, long[] Strides)[]
            {
                (np.broadcast_to(f, new Shape(2, 3, 4)), new long[] { 12, 1, 3 }),
                (np.broadcast_to(np.expand_dims(f, 1), new Shape(3, 5, 4)), new long[] { 1, 12, 3 }),
            };
            foreach (var (w, weight) in Weights)
                foreach (var (x, strides) in cases)
                {
                    var r = weight(x);
                    string label = $"{w}({string.Join(",", x.shape)})";
                    CollectionAssert.AreEqual(strides, r.Shape.Strides, $"{label}: element strides");
                    Assert.AreEqual(Hex(weight(np.ascontiguousarray(x))), Hex(r), $"{label}: values");
                }
        }

        // ---------------------------------------------------------------------------------------------
        //  Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Asserts a dump: dtype, shape, element strides (when NumPy's are listed) and bytes. The exp-based weights' float64
        ///     / float16 / complex128 values come out of the C runtime on both sides; off win-amd64 (whose ucrtbase made NumPy's
        ///     dump) each element is held to 2 ULP instead of its exact bits.
        /// </summary>
        /// <param name="c">The dump.</param>
        /// <param name="r">NumSharp's result.</param>
        private static void AssertCase(Case c, NDArray r)
        {
            Assert.AreEqual(c.Dtype, r.dtype.name, $"{c.Label}: dtype");
            CollectionAssert.AreEqual(c.Shape, r.Shape.Dimensions, $"{c.Label}: shape");
            if (c.Strides.Length > 0)
                CollectionAssert.AreEqual(c.Strides, r.Shape.Strides, $"{c.Label}: element strides");
            bool hostLibm = c.Weight is "lag" or "herm" or "herme" && c.Dtype is "float64" or "float16" or "complex128";
            if (!hostLibm || OperatingSystem.IsWindows())
            {
                Assert.AreEqual(c.Hex, Hex(r), $"{c.Label}: bytes");
                return;
            }
            byte[] expected = Convert.FromHexString(c.Hex), actual = Convert.FromHexString(Hex(r));
            int width = c.Dtype == "float16" ? 2 : 8;
            for (int i = 0; i < expected.Length; i += width)
            {
                long e = width == 2 ? BitConverter.ToInt16(expected, i) : BitConverter.ToInt64(expected, i);
                long a = width == 2 ? BitConverter.ToInt16(actual, i) : BitConverter.ToInt64(actual, i);
                Assert.IsTrue(UlpDistance(a, e) <= 2, $"{c.Label}: element at byte {i} differs by more than 2 ULP");
            }
        }

        /// <summary>The distance in units in the last place between two same-width IEEE bit patterns read as signed
        ///     integers (NaN to NaN is 0; a sign difference is the sum of both distances from zero).</summary>
        /// <param name="a">One bit pattern.</param>
        /// <param name="b">The other.</param>
        /// <returns>The distance.</returns>
        private static long UlpDistance(long a, long b)
        {
            if (a == b)
                return 0;
            static long Ordered(long v) => v < 0 ? long.MinValue - v : v;   // two's-complement order of sign-magnitude
            long d = Ordered(a) - Ordered(b);
            return d < 0 ? -d : d;
        }

        /// <summary>The logical-C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array (any layout).</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a) => Convert.ToHexString(a.tobytes()).ToLowerInvariant();

        /// <summary>The Python exception type NumSharp's exception stands for (the dumps record NumPy's).</summary>
        /// <param name="e">The exception.</param>
        /// <returns><c>TypeError</c>, <c>OverflowError</c>, <c>ValueError</c>, or the CLR name.</returns>
        private static string PythonName(Exception e) => e switch
        {
            TypeError => "TypeError",
            OverflowException => "OverflowError",
            ValueError => "ValueError",
            _ => e.GetType().Name,
        };

        /// <summary>Runs <paramref name="call"/> and returns the exception it throws.</summary>
        /// <param name="call">The call.</param>
        /// <param name="label">The case, for the failure message.</param>
        /// <returns>The exception.</returns>
        /// <exception cref="AssertFailedException">The call returned.</exception>
        private static Exception AssertThrowsAny(Action call, string label)
        {
            try
            {
                call();
            }
            catch (Exception e)
            {
                return e;
            }
            Assert.Fail($"{label}: expected an exception");
            return null;
        }

        /// <summary>Asserts that <paramref name="call"/> raises a <see cref="TypeError"/> with exactly <paramref name="text"/>.</summary>
        /// <param name="call">The call.</param>
        /// <param name="text">NumPy's text.</param>
        /// <param name="label">The case, for the failure message.</param>
        private static void AssertTypeError(Func<object> call, string text, string label)
        {
            var e = AssertThrowsAny(() => call(), label);
            Assert.IsInstanceOfType(e, typeof(TypeError), $"{label}: {e.GetType().Name}: {e.Message}");
            Assert.AreEqual(text, e.Message, label);
        }
    }
}
