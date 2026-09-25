using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> evaluation family — <c>{p}val</c>, <c>{p}val2d</c>, <c>{p}val3d</c>,
    ///     <c>{p}grid2d</c>, <c>{p}grid3d</c> and <c>{p}valnd</c> for the six bases. Every expected byte pattern
    ///     was produced by NumPy 2.4.2 (the probe inputs are spelled out next to each test), so these tests pin
    ///     BIT parity, not closeness; the broad matrix is the <c>polyeval.jsonl</c> oracle tier.
    /// </summary>
    [TestClass]
    public class PolynomialEvaluationTests
    {
        private static readonly Dictionary<string, Func<object, NDArray, bool, NDArray>> Val = new()
        {
            ["polyval"] = np.polynomial.polynomial.polyval,
            ["chebval"] = np.polynomial.chebyshev.chebval,
            ["legval"] = np.polynomial.legendre.legval,
            ["lagval"] = np.polynomial.laguerre.lagval,
            ["hermval"] = np.polynomial.hermite.hermval,
            ["hermeval"] = np.polynomial.hermite_e.hermeval,
        };

        /// <summary>Asserts dtype, shape and every byte (logical C order) against a NumPy hex dump.</summary>
        private static void AssertBytes(NDArray actual, string dtype, long[] shape, string hex, string because = "")
        {
            actual.dtype.name.Should().Be(dtype, because);
            actual.shape.Should().Equal(shape, because);
            using var c = np.ascontiguousarray(actual);
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            Convert.ToHexString(bytes).ToLowerInvariant().Should().Be(hex, because);
        }

        /// <summary>np.linspace(-1, 1, 7), the probe x of the 1-D cases.</summary>
        private static NDArray X7 => np.linspace(-1.0, 1.0, 7);

        [TestMethod]
        public void Val_Float64_MatchesNumPyBitForBit_AllBases()
        {
            // x = np.linspace(-1, 1, 7); c = [1.5, -2.25, 0.75, 3.0, -1.0]
            var expected = new Dictionary<string, string>
            {
                ["polyval"] = "000000000000e03f45cac0d3adf901408845cac0d3ad0140000000000000f83fe88745cac0d3ed3fa65bf3c32265f03f0000000000000040",
                ["chebval"] = "000000000000e03f33f0746b7e581940766b7e58a40c1040000000000000d0bf4fb7e68745ca04c0ca2265e0e9d6f8bf0000000000000040",
                ["legval"] = "000000000000e03f78ba353f2c5211408a45cac0d3ad0940000000000000e83f6ecd0f8b9481e7bfd32265e0e9d6ccbf0000000000000040",
                ["lagval"] = "acaaaaaaaaaa1f4035b55cb8ab6f1640980b77f54df20c40ffffffffffffff3f5edf24efaf83e93f6018b3d2e811a53f000000000000d0bf",
                ["hermval"] = "0000000000c04340aa5bf3c322653b407aba353f2c52184000000000000028c0f0746b7e582433c09781a75bf3c328c00000000000001a40",
                ["hermeval"] = "0000000000802740d00f8b9481a71c40e09a1f162903014000000000000002c059a40c3cdd9a14c06c7e58a40c3c18c000000000000013c0",
            };
            var c = np.array(new[] { 1.5, -2.25, 0.75, 3.0, -1.0 });
            foreach (var (name, f) in Val)
                AssertBytes(f(X7, c, true), "float64", new long[] { 7 }, expected[name], name);
        }

        [TestMethod]
        public void Val_Float32Series_AtFloat64X_RunsTheFirstStepsInFloat32()
        {
            // NumPy starts Clenshaw's accumulators at the SERIES dtype, so the first one or two steps are
            // float32 arithmetic: pre-casting c to float64 changes chebval/hermval 7/7 and hermeval 5/7.
            // x = np.linspace(-1, 1, 7); c = np.array([0.1, -0.7, 0.3, 1.9, -1.3], np.float32)
            var expected = new Dictionary<string, string>
            {
                ["polyval"] = "000000c3cccc00c0b85bf32323a8bebf756b7ec090efd13f000000a09999b93f746b7e183d63a7bfe087456a9ca5b23f000000683333d33f",
                ["chebval"] = "000000cbcccc00c06b7e582189cb0a40089e6e578021f73f000000e6fffff7bf6fcd0f366f1002c09494813395c4e5bf000000283333d33f",
                ["legval"] = "000000c3cccc00c00b3cdda4b6a3fa3f4b19786e34b9ef3f0000002c3333e1bfead6fc361460f0bf9d81a733273ad0bfffffff673333d33f",
                ["lagval"] = "55555501bbbbe9bfecf9d7c17ea77cbf04ec6074a1fcd33f010000683333d33fb0e5c2fd3b1cb63fc9e811459171cbbf0000002c2222e0bf",
                ["hermval"] = "0000d08b99d941409c1fb6160ab03340e8c3222b2c95fcbf00006086991930c05df3233338cb30c058cac0a6a5920cc00000a01b33b33140",
                ["hermeval"] = "000080c2cccc1c40ecd6fc6d479307403c2c529c0a73f2bf00008059666610c09e6e4d5db51d15c0feb0c8aa8bd711c0000000d6ccccfcbf",
            };
            var c = np.array(new[] { 0.1f, -0.7f, 0.3f, 1.9f, -1.3f });
            foreach (var (name, f) in Val)
                AssertBytes(f(X7, c, true), "float64", new long[] { 7 }, expected[name], name);
        }

        [TestMethod]
        public void Val_PythonScalarX_IsWeak_ZeroDArrayX_IsStrong()
        {
            // {p}val(0.3, float32 c) keeps float32 (a Python float adopts the partner's dtype, NEP 50);
            // {p}val(np.array(0.3), float32 c) is float64 (a 0-d array is a strong operand).
            var weak = new Dictionary<string, string>
            {
                ["polyval"] = "5ef92cbd", ["chebval"] = "4fc713c0", ["legval"] = "bd4185bf",
                ["lagval"] = "3444e83d", ["hermval"] = "12278bc1", ["hermeval"] = "e201a8c0",
            };
            var strong = new Dictionary<string, string>
            {
                ["polyval"] = "a4c64b772b9fa5bf", ["chebval"] = "b7f3fdebe97802c0", ["legval"] = "295c0fb137a8f0bf",
                ["lagval"] = "fc53e33d8508bd3f", ["hermval"] = "5ebaa95de26431c0", ["hermeval"] = "b6f37d3e3c0015c0",
            };
            var c = np.array(new[] { 0.1f, -0.7f, 0.3f, 1.9f, -1.3f });
            foreach (var (name, f) in Val)
            {
                AssertBytes(f(0.3, c, true), "float32", Array.Empty<long>(), weak[name], name + " weak");
                AssertBytes(f(np.array(0.3), c, true), "float64", Array.Empty<long>(), strong[name], name + " strong");
            }
        }

        [TestMethod]
        public void Val_ComplexX_ReproducesNumPysScalarMathAndUfuncMix()
        {
            // With a 1-D series every value is a NumPy scalar: a product touching the 0-d x ARRAY is a ufunc
            // (fused simd_cmul), every other product is scalar math (the naive formula). A Python complex x
            // makes them ALL scalar math; an array x makes them all ufuncs. The three results differ in the
            // last bit, per basis, exactly as NumPy's do.
            // cc = [1.25-0.5j, -0.75+2j, 0.3+0.1j, -1.1-0.9j, 0.45+0.65j]; x = 0.7-0.35j
            var cc = np.array(new[]
            {
                new Complex(1.25, -0.5), new Complex(-0.75, 2.0), new Complex(0.3, 0.1),
                new Complex(-1.1, -0.9), new Complex(0.45, 0.65),
            });
            var x = new Complex(0.7, -0.35);
            var zeroD = new Dictionary<string, string>
            {
                ["polyval"] = "da8f14916109f43ff8449e245d37f43f", ["chebval"] = "7ac3b645994de63f498ac8b08a070040",
                ["legval"] = "6ea301bc8541f03fefc9c3422d5ff93f", ["lagval"] = "ee6987bf2652e63ffa304278b44bd7bf",
                ["hermval"] = "6b861bf0f95111c0026a6ad95ac72740", ["hermeval"] = "fe9fc37c793b0540bc57ad4cf8e70d40",
            };
            var weak = new Dictionary<string, string>
            {
                ["polyval"] = "db8f14916109f43ffa449e245d37f43f", ["chebval"] = "7ac3b645994de63f488ac8b08a070040",
                ["legval"] = "6ea301bc8541f03ff0c9c3422d5ff93f", ["lagval"] = "ee6987bf2652e63ffa304278b44bd7bf",
                ["hermval"] = "6b861bf0f95111c0026a6ad95ac72740", ["hermeval"] = "fe9fc37c793b0540bc57ad4cf8e70d40",
            };
            var arr = new Dictionary<string, string>
            {
                ["polyval"] = "da8f14916109f43ff8449e245d37f43f", ["chebval"] = "7ac3b645994de63f498ac8b08a070040",
                ["legval"] = "6ea301bc8541f03fefc9c3422d5ff93f", ["lagval"] = "ed6987bf2652e63ffa304278b44bd7bf",
                ["hermval"] = "6b861bf0f95111c0026a6ad95ac72740", ["hermeval"] = "fe9fc37c793b0540bc57ad4cf8e70d40",
            };
            foreach (var (name, f) in Val)
            {
                AssertBytes(f(np.array(x), cc, true), "complex128", Array.Empty<long>(), zeroD[name], name + " 0-d");
                AssertBytes(f(x, cc, true), "complex128", Array.Empty<long>(), weak[name], name + " weak");
                AssertBytes(f(np.array(new[] { x }), cc, true), "complex128", new long[] { 1 }, arr[name], name + " array");
            }
        }

        [TestMethod]
        public void Val_Float16_NaNPriority_IsTheWheelsOnBothTheVectorAndTheScalarPath()
        {
            // hermeval/chebval with x = NaN(0x7e22) x16 and c = [NaN(0x7e11), 1, 2] (float16): the final add meets
            // two distinct NaNs and NumPy's HALF add keeps the SECOND operand's (0x7e22). 16 contiguous lanes run
            // the float16 vector chains; the transposed-N-D-coefficient form below runs scalar chains. Both must
            // give NumPy's bits (the oracle tier compares float NaNs as NaN, so this test is the payload gate).
            var xh = np.zeros(new Shape(16), NPTypeCode.Half);
            var ch = np.array(new[] { (Half)0.0, (Half)1.0, (Half)2.0 });
            unsafe
            {
                var px = (ushort*)xh.Storage.Address;
                for (int i = 0; i < 16; i++) px[i] = 0x7E22;
                *(ushort*)ch.Storage.Address = 0x7E11;
            }
            foreach (var f in new[] { Val["hermeval"], Val["chebval"] })
            {
                using var vec = f(xh, ch, true);
                AssertBytes(vec, "float16", new long[] { 16 }, string.Concat(Enumerable.Repeat("227e", 16)), "vector path");
                // Scalar chains: a strided x against 2-D coefficients (never buffered).
                using var c2 = np.expand_dims(ch, 1);
                using var scal = f(xh["::2"], c2, true);
                AssertBytes(scal, "float16", new long[] { 1, 8 }, string.Concat(Enumerable.Repeat("227e", 8)), "scalar path");
            }

            // inf*0 / inf-inf make the default NaN (0xfe00) meet an input NaN (0x7e33): x = +inf x16,
            // c = [NaN(0x7e33), 1, 1].
            var xi = np.full(new Shape(16), Half.PositiveInfinity, NPTypeCode.Half);
            var ci = np.array(new[] { (Half)0.0, (Half)1.0, (Half)1.0 });
            unsafe { *(ushort*)ci.Storage.Address = 0x7E33; }
            var infMix = new Dictionary<string, string>
            {
                ["polyval"] = "00fe", ["chebval"] = "337e", ["legval"] = "337e",
                ["lagval"] = "337e", ["hermval"] = "337e", ["hermeval"] = "337e",
            };
            foreach (var (name, f) in Val)
                AssertBytes(f(xi, ci, true), "float16", new long[] { 16 }, string.Concat(Enumerable.Repeat(infMix[name], 16)), name);
        }

        [TestMethod]
        public void Val_Int8X_WrapsItsX2_LikeNumPy()
        {
            // chebval(int8 [100], [0, 0, 1]): x2 = 2*x is an int8 op and wraps to -56 -> -5601.0.
            AssertBytes(np.polynomial.chebyshev.chebval(np.array(new sbyte[] { 100 }), np.array(new[] { 0, 0, 1 }), true),
                "float64", new long[] { 1 }, "0000000000e1b5c0");
        }

        [TestMethod]
        public void Val_TensorTrueAndFalse_ShapesAndValues()
        {
            var c2 = np.arange(6.0).reshape(3, 2);
            // chebval([0.5, -1, 2, 0.25], c2) -> (2, 4); chebval([0.5, -1], c2, tensor=False) -> (2,)
            AssertBytes(np.polynomial.chebyshev.chebval(np.array(new[] { 0.5, -1.0, 2.0, 0.25 }), c2),
                "float64", new long[] { 2, 4 },
                "000000000000f0bf0000000000000040000000000000404000000000000008c000000000000000000000000000000840000000000000454000000000000005c0");
            AssertBytes(np.polynomial.chebyshev.chebval(np.array(new[] { 0.5, -1.0 }), c2, tensor: false),
                "float64", new long[] { 2 }, "000000000000f0bf0000000000000840");
        }

        [TestMethod]
        public void Val_ShapeIsPreserved_AndEmptyXGivesEmptyFloat64()
        {
            // NumPy's TestEvaluation: the result has x's shape for 1-, 2- and 3-coefficient series.
            foreach (var (name, f) in Val)
            {
                for (int i = 0; i < 3; i++)
                {
                    var dims = Enumerable.Repeat(2L, i).ToArray();
                    var x = np.zeros(new Shape(dims));
                    f(x, np.array(new[] { 1.0 }), true).shape.Should().Equal(dims, name);
                    f(x, np.array(new[] { 1.0, 0.0 }), true).shape.Should().Equal(dims, name);
                    f(x, np.array(new[] { 1.0, 0.0, 0.0 }), true).shape.Should().Equal(dims, name);
                }
                var empty = f(np.array(Array.Empty<double>()), np.array(new[] { 1.0 }), true);
                empty.size.Should().Be(0);
                empty.dtype.name.Should().Be("float64");
            }
        }

        [TestMethod]
        public void Val_IntegerAndBoolCoefficients_AreEvaluatedAsFloat64()
        {
            foreach (var (name, f) in Val)
            {
                f(np.array(new[] { 1.0, 2.0 }), np.array(new[] { 1, 2, 3 }), true).dtype.name.Should().Be("float64", name);
                f(np.array(new[] { 1.0, 2.0 }), np.array(new[] { true, false, true }), true).dtype.name.Should().Be("float64", name);
            }
        }

        [TestMethod]
        public void Val_Errors_CarryNumPysTexts()
        {
            foreach (var (name, f) in Val)
            {
                Action empty = () => f(np.array(new[] { 1.0 }), np.array(Array.Empty<double>()), true);
                empty.Should().Throw<IndexError>().WithMessage(name == "polyval"
                    ? "index -1 is out of bounds for axis 0 with size 0"
                    : "index -2 is out of bounds for axis 0 with size 0");
            }

            Action bcast = () => np.polynomial.chebyshev.chebval(np.array(new[] { 1.0, 2, 3 }), np.ones(new Shape(3, 2)), tensor: false);
            bcast.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,) (3,) ");

            Action overflow = () => np.polynomial.laguerre.lagval(np.array(new sbyte[] { 1, 2 }), np.ones(new Shape(70)));
            overflow.Should().Throw<OverflowException>().WithMessage("Python integer 137 out of bounds for int8");

            Action v2 = () => np.polynomial.chebyshev.chebval2d(np.ones(new Shape(3)), np.ones(new Shape(2)), np.ones(new Shape(2, 2)));
            v2.Should().Throw<ValueError>().WithMessage("x, y are incompatible");
            Action v3 = () => np.polynomial.chebyshev.chebval3d(np.ones(new Shape(3)), np.ones(new Shape(3)), np.ones(new Shape(2)), np.ones(new Shape(2, 2, 2)));
            v3.Should().Throw<ValueError>().WithMessage("x, y, z are incompatible");
            Action v4 = () => np.polynomial.chebyshev.chebvalnd(
                new[] { np.ones(new Shape(3)), np.ones(new Shape(3)), np.ones(new Shape(3)), np.ones(new Shape(2)) }, np.ones(new Shape(2, 2, 2, 2)));
            v4.Should().Throw<ValueError>().WithMessage("ordinates are incompatible");
            Action v0 = () => np.polynomial.chebyshev.chebvalnd(Array.Empty<NDArray>(), np.ones(new Shape(2, 2)));
            v0.Should().Throw<IndexError>().WithMessage("list index out of range");
        }

        [TestMethod]
        public void Grid2d_PythonScalarOrdinates_StayWeak()
        {
            // _gridnd hands its ordinates to {p}val unconverted: a Python float stays weak (float32 c stays
            // float32), unlike val2d's np.asanyarray.
            var cg = np.array(new[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } });
            AssertBytes(np.polynomial.chebyshev.chebgrid2d(0.5, np.array(new[] { 1.0, 2.0 }), cg),
                "float64", new long[] { 2 }, "0000000000001a400000000000002540");
            AssertBytes(np.polynomial.chebyshev.chebgrid2d(0.5, 0.25, cg), "float32", Array.Empty<long>(), "00006040");
        }

        [TestMethod]
        public void Val_EveryLayout_EqualsTheContiguousCopy()
        {
            // The kernel reads x through NDIter (buffered for a strided x in the 1-D form); every layout must
            // produce exactly the contiguous copy's bytes.
            var m = np.linspace(-1.5, 1.5, 48).reshape(6, 8);
            var views = new[] { m, m.T, m[":, ::2"], m["::-1"], m["1:5, 2:7"], np.asfortranarray(m), np.broadcast_to(m["0"], new Shape(6, 8)) };
            var c = np.array(new[] { 0.3, -1.2, 0.7, 2.1, -0.4, 0.9 });
            foreach (var (name, f) in Val)
                foreach (var v in views)
                {
                    using var got = f(v, c, true);
                    using var want = f(v.copy(), c, true);
                    np.array_equal(got, want).Should().BeTrue($"{name} on shape ({string.Join(",", v.shape)})");
                }
        }

        [TestMethod]
        public void Val_DecimalAndChar_FollowTheSameRecurrence()
        {
            // No NumPy analog: Decimal evaluates NumPy's step lines in System.Decimal (the independent-oracle
            // convention), Char behaves as the uint16 it stores.
            var xd = np.array(new[] { 0.5m, -0.25m, 1.75m });
            var cd = np.array(new[] { 1.5m, -2m, 0.75m, 3m });
            decimal Clenshaw(decimal x)
            {
                // chebval: x2 = 2*x ; c0 = c[-i] - c1 ; c1 = tmp + c1*x2 ; return c0 + c1*x
                decimal[] c = { 1.5m, -2m, 0.75m, 3m };
                decimal x2 = 2 * x, c0 = c[2], c1 = c[3];
                for (int i = 3; i <= 4; i++) { var tmp = c0; c0 = c[4 - i] - c1; c1 = tmp + c1 * x2; }
                return c0 + c1 * x;
            }
            var got = np.polynomial.chebyshev.chebval(xd, cd);
            got.typecode.Should().Be(NPTypeCode.Decimal);
            for (int i = 0; i < 3; i++)
                got.GetAtIndex<decimal>(i).Should().Be(Clenshaw(new[] { 0.5m, -0.25m, 1.75m }[i]));

            var xc = np.array(new[] { 'a', 'b', (char)7 });
            var xu = np.array(new ushort[] { 'a', 'b', 7 });
            var cf = np.array(new[] { 0.5, -1.0, 0.25 });
            foreach (var (name, f) in Val)
                np.array_equal(f(xc, cf, true), f(xu, cf, true)).Should().BeTrue(name);
        }

        /// <summary>The raw bytes of a result in logical C order (dtype and shape asserted by the caller).</summary>
        private static byte[] RawBytes(NDArray a)
        {
            using var c = np.ascontiguousarray(a);
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            return bytes;
        }

        /// <summary>45 points of every dtype (the shared input shape of the lane tests below: 32 + 8 + 5 at 8 lanes,
        ///     32 + 12 + 1 at 4, 40 + 4 + 1 at 2 — the U-chain stage, the 1-chain stage and the scalar tail).</summary>
        private static NDArray Points(NPTypeCode t) => t switch
        {
            NPTypeCode.Boolean => (np.arange(45) % 2).astype(NPTypeCode.Boolean),
            NPTypeCode.Complex => (np.linspace(-1.2, 1.3, 45) * new Complex(0.8, -0.35)).astype(NPTypeCode.Complex),
            NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => np.linspace(-1.2, 1.3, 45).astype(t),
            _ => (np.arange(45) % 7).astype(t),   // 0..6: every integer dtype, unsigned included
        };

        [TestMethod]
        public void Val_EveryDtypePair_HasVectorLanes_NoScalarFallback()
        {
            // Every NumPy x dtype meets every inexact series dtype, as a 1-D series and one series per point:
            // the lane table (ILKernelGenerator.Polynomial.Lanes.cs) must hold every per-point value those
            // recurrences produce, or the kernel compile silently drops to scalar chains (same bits, 2-6x
            // slower) — which PolyVectorFallbacks counts. Decimal/Char ride along to prove they never throw.
            long before = NumSharp.Backends.Kernels.ILKernelGenerator.PolyVectorFallbacks;
            var xTypes = new[] { NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half,
                NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex };
            var cTypes = new[] { NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex };
            foreach (var (name, f) in Val)
                foreach (var tx in xTypes)
                    foreach (var tc in cTypes)
                    {
                        using var x = Points(tx);
                        using var c1 = np.linspace(-0.9, 1.1, 6).astype(tc);
                        using var cp = np.linspace(-0.9, 1.1, 6 * 45).reshape(6, 45).astype(tc);
                        f(x, c1, true).Dispose();
                        f(x, cp, false).Dispose();
                    }
            (NumSharp.Backends.Kernels.ILKernelGenerator.PolyVectorFallbacks - before).Should().Be(0,
                "every NumPy dtype pair has a vector lane kind and conversion");
        }

        [TestMethod]
        public void Val_VectorAndScalarParts_AgreeByteForByte()
        {
            // The same per-point series read contiguously (vector part: lanes of every dtype involved) and
            // through a column-strided view (scalar part): the two kernels' parts must produce identical bytes
            // — NaN payloads and signed zeros included — for every dtype pair and basis.
            var xTypes = new[] { NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Half,
                NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex };
            var cTypes = new[] { NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex };
            foreach (var (name, f) in Val)
                foreach (var tx in xTypes)
                    foreach (var tc in cTypes)
                    {
                        using var x = Points(tx);
                        using var wide = np.linspace(-0.9, 1.1, 6 * 90).reshape(6, 90).astype(tc);
                        var strided = wide[":, ::2"];                 // scalar per-point part (column stride 2)
                        using var dense = strided.copy();              // vector per-point part
                        using var got = f(x, strided, false);
                        using var want = f(x, dense, false);
                        got.dtype.Should().Be(want.dtype, $"{name} x={tx} c={tc}");
                        RawBytes(got).Should().Equal(RawBytes(want), $"{name} x={tx} c={tc}");
                    }
        }

        [TestMethod]
        public void Val_IsThreadSafe_UnderConcurrentFirstUse()
        {
            // Concurrent first calls must compile ONE kernel per key and leave the constant pools consistent.
            var x = np.linspace(-1.0, 1.0, 1001);
            var c = np.array(new[] { 0.1f, -0.7f, 0.3f, 1.9f, -1.3f, 0.6f, -0.2f });
            var want = Val.ToDictionary(kv => kv.Key, kv => kv.Value(x, c, true));
            Parallel.For(0, 64, i =>
            {
                foreach (var (name, f) in Val)
                {
                    using var got = f(x, c, true);
                    if (!np.array_equal(got, want[name]))
                        throw new InvalidOperationException($"{name} diverged on thread {i}");
                }
            });
        }
    }
}
