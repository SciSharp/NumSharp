using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> Vandermonde family (plan U5): <c>{p}vander</c>, <c>{p}vander2d</c> and <c>{p}vander3d</c> of the
    ///     six bases. Three kinds of test live here: NumPy 2.4.2's own <c>TestVander</c> suites ported (a column is the basis
    ///     polynomial, a 2-D / 3-D matrix times the flattened coefficients is <c>{p}val2d</c> / <c>{p}val3d</c>); byte dumps
    ///     NumPy produced (the probe inputs are spelled out next to each test — the signalling-NaN and NaN-pair dumps pin the
    ///     NaN BITS the byte oracle tokenizes); and what the oracle (<c>polyvander.jsonl</c>) cannot see — error texts NumPy
    ///     raises machine-dependently (MemoryError), the str / object refusals, decimal and char, the int / object overloads,
    ///     the kernel's block loop driven with tiny blocks, and its vector lanes.
    /// </summary>
    [TestClass]
    public class PolynomialVanderTests
    {
        /// <summary>One basis's Vandermonde family plus its evaluators.</summary>
        /// <param name="Name">The basis prefix NumPy uses (<c>poly</c>, <c>cheb</c>, ...).</param>
        /// <param name="Vander"><c>{p}vander(x, deg)</c> — the object overload.</param>
        /// <param name="VanderInt"><c>{p}vander(x, deg)</c> — the int overload.</param>
        /// <param name="Vander2d"><c>{p}vander2d(x, y, deg)</c>.</param>
        /// <param name="Vander3d"><c>{p}vander3d(x, y, z, deg)</c>.</param>
        /// <param name="Val"><c>{p}val(x, c, tensor)</c>.</param>
        /// <param name="Val2d"><c>{p}val2d(x, y, c)</c>.</param>
        /// <param name="Val3d"><c>{p}val3d(x, y, z, c)</c>.</param>
        private sealed record Basis(string Name, Func<object, object, NDArray> Vander, Func<object, int, NDArray> VanderInt,
            Func<object, object, object, NDArray> Vander2d, Func<object, object, object, object, NDArray> Vander3d,
            Func<object, object, bool, NDArray> Val, Func<object, object, object, NDArray> Val2d,
            Func<object, object, object, object, NDArray> Val3d);

        /// <summary>The six bases, in NumPy's module order.</summary>
        private static readonly Basis[] Bases =
        {
            new("poly", np.polynomial.polynomial.polyvander, np.polynomial.polynomial.polyvander, np.polynomial.polynomial.polyvander2d,
                np.polynomial.polynomial.polyvander3d, np.polynomial.polynomial.polyval, np.polynomial.polynomial.polyval2d,
                np.polynomial.polynomial.polyval3d),
            new("cheb", np.polynomial.chebyshev.chebvander, np.polynomial.chebyshev.chebvander, np.polynomial.chebyshev.chebvander2d,
                np.polynomial.chebyshev.chebvander3d, np.polynomial.chebyshev.chebval, np.polynomial.chebyshev.chebval2d,
                np.polynomial.chebyshev.chebval3d),
            new("leg", np.polynomial.legendre.legvander, np.polynomial.legendre.legvander, np.polynomial.legendre.legvander2d,
                np.polynomial.legendre.legvander3d, np.polynomial.legendre.legval, np.polynomial.legendre.legval2d,
                np.polynomial.legendre.legval3d),
            new("lag", np.polynomial.laguerre.lagvander, np.polynomial.laguerre.lagvander, np.polynomial.laguerre.lagvander2d,
                np.polynomial.laguerre.lagvander3d, np.polynomial.laguerre.lagval, np.polynomial.laguerre.lagval2d,
                np.polynomial.laguerre.lagval3d),
            new("herm", np.polynomial.hermite.hermvander, np.polynomial.hermite.hermvander, np.polynomial.hermite.hermvander2d,
                np.polynomial.hermite.hermvander3d, np.polynomial.hermite.hermval, np.polynomial.hermite.hermval2d,
                np.polynomial.hermite.hermval3d),
            new("herme", np.polynomial.hermite_e.hermevander, np.polynomial.hermite_e.hermevander, np.polynomial.hermite_e.hermevander2d,
                np.polynomial.hermite_e.hermevander3d, np.polynomial.hermite_e.hermeval, np.polynomial.hermite_e.hermeval2d,
                np.polynomial.hermite_e.hermeval3d),
        };

        /// <summary>The logical-C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array (any layout).</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a)
        {
            using var c = np.ascontiguousarray(a);
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        /// <summary>Asserts dtype, shape and every byte (logical C order) against a NumPy hex dump — BIT parity.</summary>
        /// <param name="actual">The result.</param><param name="dtype">NumPy's dtype name.</param>
        /// <param name="shape">NumPy's shape.</param><param name="hex">NumPy's <c>tobytes().hex()</c>.</param>
        /// <param name="because">Context for the failure message.</param>
        private static void AssertBytes(NDArray actual, string dtype, long[] shape, string hex, string because = "")
        {
            actual.dtype.name.Should().Be(dtype, because);
            actual.shape.Should().Equal(shape, because);
            Hex(actual).Should().Be(hex, because);
        }

        /// <summary><c>[0]*i + [1]</c> — NumPy's test series (the i-th basis polynomial), as a Python list of ints.</summary>
        /// <param name="i">The degree.</param><returns>The list.</returns>
        private static object[] Unit(int i) => Enumerable.Repeat((object)0, i).Append(1).ToArray();

        /// <summary>NumPy's <c>assert_almost_equal</c> (decimal=7) over two arrays of the same shape (float64 values).</summary>
        /// <param name="actual">The values.</param><param name="desired">The expected values.</param>
        /// <param name="because">Context.</param>
        private static void AlmostEqual(NDArray actual, NDArray desired, string because)
        {
            actual.shape.Should().Equal(desired.shape, because);
            var a = actual.astype(np.float64).ToArray<double>();
            var d = desired.astype(np.float64).ToArray<double>();
            for (int i = 0; i < a.Length; i++)
                System.Math.Abs(a[i] - d[i]).Should().BeLessThan(1.5e-7, $"{because} [{i}]");
        }

        /// <summary>A NumPy error's type and verbatim text.</summary>
        /// <typeparam name="TException">The expected exception type.</typeparam>
        /// <param name="call">The call.</param><param name="text">NumPy's message.</param>
        private static void Raises<TException>(Action call, string text) where TException : Exception
            => call.Should().Throw<TException>().Which.Message.Should().StartWith(text);

        // =========================================================================================================
        //  NumPy 2.4.2's own TestVander (numpy/polynomial/tests/test_{polynomial,chebyshev,...}.py), ported
        // =========================================================================================================

        /// <summary>
        ///     <c>test_{p}vander</c>: <c>{p}vander(x, 3)</c> of a 1-D and a 2-D x has shape <c>x.shape + (4,)</c>, and its column i
        ///     is the basis polynomial i — <c>{p}val(x, [0]*i + [1])</c>.
        /// </summary>
        [TestMethod]
        public void Vander_ColumnIIsTheBasisPolynomialI_NumPyTestVander()
        {
            foreach (var b in Bases)
            {
                var x = np.arange(3);
                var v = b.Vander(x, 3);
                v.shape.Should().Equal(new long[] { 3, 4 }, b.Name);
                for (int i = 0; i < 4; i++)
                    AlmostEqual(v[$"..., {i}"], b.Val(x, Unit(i), true), $"{b.Name} 1-D column {i}");

                var x2 = np.array(new long[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
                var v2 = b.Vander(x2, 3);
                v2.shape.Should().Equal(new long[] { 3, 2, 4 }, b.Name);
                for (int i = 0; i < 4; i++)
                    AlmostEqual(v2[$"..., {i}"], b.Val(x2, Unit(i), true), $"{b.Name} 2-D column {i}");
            }
        }

        /// <summary>
        ///     <c>test_{p}vander2d</c> / <c>test_{p}vander3d</c>: the matrix times the flattened coefficients is <c>{p}val2d</c> /
        ///     <c>{p}val3d</c> at the points (non-square coefficient arrays), and a list-wrapped point set keeps its leading axis.
        /// </summary>
        [TestMethod]
        public void Vander2dVander3d_TimesFlatCoefficients_AreVal2dVal3d_NumPyTestVander()
        {
            np.random.seed(7);
            var x = np.random.random(new Shape(3, 5)) * 2 - 1;
            NDArray x1 = x["0"], x2 = x["1"], x3 = x["2"];
            foreach (var b in Bases)
            {
                var c = np.random.random(new Shape(2, 3));
                var van = b.Vander2d(x1, x2, new[] { 1, 2 });
                AlmostEqual(np.dot(van, c.flat), b.Val2d(x1, x2, c), $"{b.Name}vander2d");
                b.Vander2d(new object[] { x1 }, new object[] { x2 }, new[] { 1, 2 }).shape.Should().Equal(new long[] { 1, 5, 6 }, b.Name);

                var c3 = np.random.random(new Shape(2, 3, 4));
                var van3 = b.Vander3d(x1, x2, x3, new[] { 1, 2, 3 });
                AlmostEqual(np.dot(van3, c3.flat), b.Val3d(x1, x2, x3, c3), $"{b.Name}vander3d");
                b.Vander3d(new object[] { x1 }, new object[] { x2 }, new object[] { x3 }, new[] { 1, 2, 3 }).shape
                    .Should().Equal(new long[] { 1, 5, 24 }, b.Name);
            }
        }

        /// <summary><c>test_polyvandernegdeg</c>: a negative degree is <c>ValueError("deg must be non-negative")</c> — for every basis.</summary>
        [TestMethod]
        public void Vander_NegativeDegree_Raises_NumPyTestVander()
        {
            foreach (var b in Bases)
            {
                Raises<ValueError>(() => b.VanderInt(np.arange(3), -1), "deg must be non-negative");
                Raises<ValueError>(() => b.Vander(np.arange(3), -1L), "deg must be non-negative");
            }
        }

        // =========================================================================================================
        //  Byte dumps of NumPy 2.4.2
        // =========================================================================================================

        /// <summary>
        ///     One dump per dtype family and form (NumPy's inputs spelled out): float64 chebvander, float32 legvander, float16
        ///     hermvander (the HALF loop per op), complex128 lagvander (simd_cmul and Smith's division by i), int16 legvander (float64),
        ///     float64 hermevander2d from Python lists, complex128 polyvander3d of complex Python scalars.
        /// </summary>
        [TestMethod]
        public void Vander_EveryDtypeFamily_MatchesNumPyBitForBit()
        {
            // np.polynomial.chebyshev.chebvander([0.1, -0.7, 1.3], 4)
            AssertBytes(np.polynomial.chebyshev.chebvander(new object[] { 0.1, -0.7, 1.3 }, 4), "float64", new long[] { 3, 5 },
                "000000000000f03f9a9999999999b93f5c8fc2f5285cefbf8c6ce7fba9f1d2bf8104c58f3177ed3f000000000000f03f666666666666e6bfa014ae47e17a94bfb39defa7c64be73fef38454772f9efbf000000000000f03fcdccccccccccf43f0bd7a3703d0a03405c643bdf4f8d134082d93d7958a82440");
            // np.polynomial.legendre.legvander(np.array([0.1, -0.7, 1.3], np.float32), 5)
            AssertBytes(np.polynomial.legendre.legvander(np.array(new float[] { 0.1f, -0.7f, 1.3f }), 5), "float32", new long[] { 3, 6 },
                "0000803fcdcccc3dec51f8be3d0a17be2506ad3ee21e373e0000803f333333bfd4a3703ebb1e453edcf9d2be55fbba3e0000803f6666a63f703d024050b86240d20dd140ca3f4741");
            // np.polynomial.hermite.hermvander(np.array([0.1, -0.7, 1.3], np.float16), 5)
            AssertBytes(np.polynomial.hermite.hermvander(np.array(new double[] { 0.1, -0.7, 1.3 }).astype(np.float16), 5), "float16", new long[] { 3, 6 },
                "003c6632d7bfc4bcc249eb49003c9abd00a9a845b0c750d0003c3341c244e03fdccdcbd4");
            // np.polynomial.laguerre.lagvander(np.array([0.3+0.7j, -1.1+0.2j, 2.3-0.9j]), 4)
            AssertBytes(np.polynomial.laguerre.lagvander(np.array(new Complex[] { new(0.3, 0.7), new(-1.1, 0.2), new(2.3, -0.9) }), 4),
                "complex128", new long[] { 3, 5 },
                "000000000000f03f0000000000000000666666666666e63f666666666666e6bf9c9999999999c93f0ad7a3703d0af3bf2cdd24068195dbbf8207f344fd1bf7bfee5d70f3b3fef1bf6a2e244da938f7bf000000000000f03f0000000000000000cdcccccccccc00409a9999999999c9bf48e17a14ae470e40d7a3703d0ad7e3bfac638207f3041940456fcb5a1d13f6bffc81987140842340e84b5cb4ed2105c0000000000000f03f0000000000000000ccccccccccccf4bfcdccccccccccec3fc3f5285c8fc2f5bf44e17a14ae47d1bf82865d0172afd1bf6abc74931804f4bf9a39fe1ecf41ef3f842f4ca60a46f5bf");
            // np.polynomial.legendre.legvander(np.array([3, -2], np.int16), 3)
            AssertBytes(np.polynomial.legendre.legvander(np.array(new short[] { 3, -2 }), 3), "float64", new long[] { 2, 4 },
                "000000000000f03f00000000000008400000000000002a400000000000804f40000000000000f03f00000000000000c0000000000000164000000000000031c0");
            // np.polynomial.hermite_e.hermevander2d([0.2, -0.4], [1.5, 0.3], [2, 1])
            AssertBytes(np.polynomial.hermite_e.hermevander2d(new object[] { 0.2, -0.4 }, new object[] { 1.5, 0.3 }, new[] { 2, 1 }),
                "float64", new long[] { 2, 6 },
                "000000000000f03f000000000000f83f9a9999999999c93f343333333333d33fb81e85eb51b8eebf0ad7a3703d0af7bf000000000000f03f333333333333d33f9a9999999999d9bfb81e85eb51b8bebfe17a14ae47e1eabf54e3a59bc420d0bf");
            // np.polynomial.polynomial.polyvander3d([0.5+0.1j], [0.25-1j], [2+0.5j], [1, 1, 2])
            AssertBytes(np.polynomial.polynomial.polyvander3d(new object[] { new Complex(0.5, 0.1) }, new object[] { new Complex(0.25, -1) },
                    new object[] { new Complex(2, 0.5) }, new[] { 1, 1, 2 }), "complex128", new long[] { 1, 12 },
                "000000000000f03f00000000000000000000000000000040000000000000e03f0000000000000e400000000000000040000000000000d03f000000000000f0bf000000000000f03f000000000000febf00000000008007400000000000000ac0000000000000e03f9a9999999999b93f666666666666ee3fcdccccccccccdc3fcdccccccccccfa3f000000000000f63fcdcccccccccccc3f666666666666debf000000000000e63fcccccccccccceabf3333333333b3fc3fcdcccccccc4cf5bf");
        }

        /// <summary>
        ///     The NaN BITS (the byte oracle compares NaN as a token): <c>x + 0.0</c> quiets a signalling NaN and keeps its
        ///     payload (float64 <c>0x7ff4…0123</c> → <c>0x7ffc…0123</c>; float16 <c>0x7d01</c> → <c>0x7f01</c>, and -0.0 reads as
        ///     +0.0), and the outer product keeps the SECOND operand's NaN when both are NaN — NumPy's FLOAT/DOUBLE_multiply on
        ///     win-amd64 at every loop length (x86's mulpd would keep the first's): polyvander2d of NaN(p1) with NaN(p2)
        ///     is NaN(p2) everywhere, and polyvander3d with a third NaN(p3) is NaN(p3).
        /// </summary>
        [TestMethod]
        public void Vander_NaNBits_QuietedPayloads_SecondOperandWinsTheProduct()
        {
            var snan64 = np.array(new ulong[] { 0x7ff4000000000123UL }).view(np.float64);
            AssertBytes(np.polynomial.polynomial.polyvander(snan64, 2), "float64", new long[] { 1, 3 },
                "230100000000fc7f230100000000fc7f230100000000fc7f");
            var snan16 = np.array(new ushort[] { 0x7d01, 0x8000 }).view(np.float16);
            AssertBytes(np.polynomial.chebyshev.chebvander(snan16, 2), "float16", new long[] { 2, 3 }, "017f017f017f003c000000bc");

            var nan1 = np.array(new ulong[] { 0x7ff8000000000001UL }).view(np.float64);
            var nan2 = np.array(new ulong[] { 0xfff8000000000002UL }).view(np.float64);
            var nan3 = np.array(new ulong[] { 0x7ff8000000000003UL }).view(np.float64);
            AssertBytes(np.polynomial.polynomial.polyvander2d(nan1, nan2, new[] { 1, 1 }), "float64", new long[] { 1, 4 },
                string.Concat(Enumerable.Repeat("020000000000f8ff", 4)));
            // Nine points: the vector loop (4 lanes) and its tail.
            AssertBytes(np.polynomial.polynomial.polyvander2d(np.repeat(nan1, 9), np.repeat(nan2, 9), new[] { 1, 1 }), "float64",
                new long[] { 9, 4 }, string.Concat(Enumerable.Repeat("020000000000f8ff", 36)));
            AssertBytes(np.polynomial.polynomial.polyvander3d(nan1, nan2, nan3, new[] { 1, 1, 1 }), "float64", new long[] { 1, 8 },
                string.Concat(Enumerable.Repeat("030000000000f87f", 8)));
            var f1 = np.array(new uint[] { 0x7fc00001 }).view(np.float32);
            var f2 = np.array(new uint[] { 0xffc00002 }).view(np.float32);
            AssertBytes(np.polynomial.polynomial.polyvander2d(np.repeat(f1, 17), np.repeat(f2, 17), new[] { 1, 1 }), "float32",
                new long[] { 17, 4 }, string.Concat(Enumerable.Repeat("0200c0ff", 68)));
        }

        // =========================================================================================================
        //  Arguments and errors, in NumPy's order
        // =========================================================================================================

        /// <summary>
        ///     <c>polyutils._as_int</c> — Python's <c>operator.index</c> — decides the degree, and its TypeError formats the refused
        ///     value as NumPy's f-string does (<c>format(x, '')</c>: a NumPy scalar / 0-d array as <c>float(x)</c>, an array as its
        ///     str, a list / tuple as its repr). Every text probed against NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void Vander_DegreeKinds_OperatorIndexAndNumPysTexts()
        {
            var P = np.polynomial.polynomial;
            var x = np.array(new[] { 0.5, -1.5, 2.0 });
            // Accepted: Python ints of any C# width, bool (True is 1), BigInteger, char (NumSharp's integer dtype), 0-d integers.
            foreach (object d in new object[] { 2, 2L, (byte)2, (ulong)2, new BigInteger(2), np.array(2), np.array((sbyte)2), np.array((ulong)2) })
                Hex(P.polyvander(x, d)).Should().Be(Hex(P.polyvander(x, 2)), $"{d.GetType().Name}");
            P.polyvander(x, true).shape.Should().Equal(new long[] { 3, 2 });
            P.polyvander(x, false).shape.Should().Equal(new long[] { 3, 1 });
            P.polyvander(x, '\u0002').shape.Should().Equal(new long[] { 3, 3 });

            string received = "deg must be an integer, received ";
            Raises<TypeError>(() => P.polyvander(x, 1.0), received + "1.0");
            Raises<TypeError>(() => P.polyvander(x, 1.5), received + "1.5");
            Raises<TypeError>(() => P.polyvander(x, -0.0), received + "-0.0");
            Raises<TypeError>(() => P.polyvander(x, 1e20), received + "1e+20");
            Raises<TypeError>(() => P.polyvander(x, double.PositiveInfinity), received + "inf");
            Raises<TypeError>(() => P.polyvander(x, double.NaN), received + "nan");
            Raises<TypeError>(() => P.polyvander(x, (Half)1000), received + "1000.0");                 // np.float16's __format__ is float(self)
            Raises<TypeError>(() => P.polyvander(x, new Complex(2, 0)), received + "(2+0j)");
            Raises<TypeError>(() => P.polyvander(x, new Complex(0, 1)), received + "1j");
            Raises<TypeError>(() => P.polyvander(x, "2"), received + "2");
            Raises<TypeError>(() => P.polyvander(x, (object)null), received + "None");
            Raises<TypeError>(() => P.polyvander(x, np.array(true)), received + "True");              // 0-d bool: no __index__
            Raises<TypeError>(() => P.polyvander(x, np.array(2.5f)), received + "2.5");
            Raises<TypeError>(() => P.polyvander(x, np.array(new Complex(2, 0))), received + "(2+0j)");
            Raises<TypeError>(() => P.polyvander(x, np.array(new[] { 2 })), received + "[2]");
            Raises<TypeError>(() => P.polyvander(x, new[] { 1, 2 }), received + "[1 2]");               // a typed C# array is an ndarray
            Raises<TypeError>(() => P.polyvander(x, new object[] { 1, "a", null, 2.5 }), received + "[1, 'a', None, 2.5]");
            Raises<TypeError>(() => P.polyvander(x, ValueTuple.Create(2)), received + "(2,)");
            Raises<TypeError>(() => P.polyvander(x, (1, 2)), received + "(1, 2)");
            Raises<TypeError>(() => P.polyvander(x, new object[] { np.array(new[] { 1.5 }) }), received + "[array([1.5])]");
            Raises<TypeError>(() => P.polyvander(x, new object[] { (Half)2 }), received + "[np.float16(2.0)]");

            // Negative / too large, NumPy's order: the sign before x, the npy_intp conversion and the byte count at np.empty.
            Raises<ValueError>(() => P.polyvander(x, -BigInteger.Pow(2, 70)), "deg must be non-negative");
            Raises<ValueError>(() => P.polyvander(x, np.array(-1)), "deg must be non-negative");
            Raises<ValueError>(() => P.polyvander(x, long.MaxValue), "Maximum allowed dimension exceeded");
            Raises<ValueError>(() => P.polyvander(x, BigInteger.Pow(2, 64)), "Maximum allowed dimension exceeded");
            Raises<ValueError>(() => P.polyvander(x, long.MaxValue - 1), "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
            Raises<ValueError>(() => P.polyvander(np.zeros(0), 1L << 62), "array is too big;");
            // np.empty((2**40 + 1, 3)): 26 TB — NumPy's MemoryError (the text is machine-dependent, so the oracle skips it).
            FluentActions.Invoking(() => P.polyvander(x, 1L << 40)).Should().Throw<OutOfMemoryException>();
        }

        /// <summary>
        ///     The degree is checked BEFORE x is converted (NumPy's first two statements), then x: a ragged list is np.array's
        ///     ValueError, a str / None / object x is refused where NumPy's <c>x + 0.0</c> fails or computes with Python
        ///     objects (NumSharp has no str / object dtype).
        /// </summary>
        [TestMethod]
        public void Vander_DegreeBeforePoints_PointsRefusedWhereNumPyFails()
        {
            var L = np.polynomial.legendre;
            Raises<ValueError>(() => L.legvander("abc", -1), "deg must be non-negative");
            Raises<TypeError>(() => L.legvander("abc", 1.5), "deg must be an integer, received 1.5");
            Raises<ValueError>(() => L.legvander(new object[] { new object[] { 1, 2 }, new object[] { 3 } }, -1), "deg must be non-negative");
            Raises<ValueError>(() => L.legvander(new object[] { new object[] { 1, 2 }, new object[] { 3 } }, 2),
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. " +
                "The detected shape was (2,) + inhomogeneous part.");
            FluentActions.Invoking(() => L.legvander("abc", 2)).Should().Throw<NotSupportedException>();
            FluentActions.Invoking(() => L.legvander((object)null, 2)).Should().Throw<NotSupportedException>();
            FluentActions.Invoking(() => L.legvander(new object[] { 1, null }, 2)).Should().Throw<NotSupportedException>();
            FluentActions.Invoking(() => L.legvander(new object[] { BigInteger.Pow(2, 70) }, 2)).Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     <c>_vander_nd</c>'s order: <c>len(deg)</c> (TypeError for a scalar, <c>len() of unsized object</c> for a 0-d array), the
        ///     count, the points' stacking (np.asarray: one shape, no broadcasting), then each dimension's degree in turn, and
        ///     for no points NumPy's reshape ValueError. The degree items are Python's <c>deg[k]</c>: a str's characters, an
        ///     ndarray's NumPy scalars (np.float64 / np.True_ refused) or rows.
        /// </summary>
        [TestMethod]
        public void VanderNd_DegreeContainersAndPoints_NumPysOrderAndTexts()
        {
            var C = np.polynomial.chebyshev;
            var a = np.array(new[] { 0.5, 0.25 });
            var b = np.array(new[] { 0.1, 0.2 });
            foreach (object deg in new object[] { new[] { 1, 2 }, new long[] { 1, 2 }, (1, 2), new object[] { 1L, 2L }, new System.Collections.Generic.List<int> { 1, 2 },
                                                  np.array(new[] { 1, 2 }), np.array(new byte[] { 1, 2 }), new object[] { true, 2 } })
            {
                var r = C.chebvander2d(a, b, deg);
                r.shape.Should().Equal(new long[] { 2, 6 }, deg.GetType().Name);
            }
            Raises<TypeError>(() => C.chebvander2d(a, b, 2), "object of type 'int' has no len()");
            Raises<TypeError>(() => C.chebvander2d(a, b, 2.0), "object of type 'float' has no len()");
            Raises<TypeError>(() => C.chebvander2d(a, b, true), "object of type 'bool' has no len()");
            Raises<TypeError>(() => C.chebvander2d(a, b, null), "object of type 'NoneType' has no len()");
            Raises<TypeError>(() => C.chebvander2d(a, b, (Half)2), "object of type 'numpy.float16' has no len()");
            Raises<TypeError>(() => C.chebvander2d(a, b, np.array(2)), "len() of unsized object");
            Raises<ValueError>(() => C.chebvander2d(a, b, new[] { 1, 1, 1 }), "Expected 2 dimensions of degrees, got 3");
            Raises<ValueError>(() => C.chebvander2d(a, b, "1"), "Expected 2 dimensions of degrees, got 1");
            Raises<ValueError>(() => C.chebvander3d(a, b, a, new[] { 1, 2 }), "Expected 3 dimensions of degrees, got 2");
            Raises<ValueError>(() => C.chebvander2d(a, b, np.array(new[,] { { 1, 2 } })), "Expected 2 dimensions of degrees, got 1");
            Raises<TypeError>(() => C.chebvander2d(a, b, "ab"), "deg must be an integer, received a");
            Raises<TypeError>(() => C.chebvander2d(a, b, "12"), "deg must be an integer, received 1");
            Raises<TypeError>(() => C.chebvander2d(a, b, new[] { 1.0, 2.0 }), "deg must be an integer, received 1.0");
            Raises<TypeError>(() => C.chebvander2d(a, b, new[] { true, false }), "deg must be an integer, received True");
            Raises<TypeError>(() => C.chebvander2d(a, b, np.array(new[,] { { 1 }, { 2 } })), "deg must be an integer, received [1]");
            Raises<TypeError>(() => C.chebvander2d(a, b, new object[] { new object[] { 1 }, new object[] { 2 } }), "deg must be an integer, received [1]");
            Raises<ValueError>(() => C.chebvander2d(a, b, new object[] { 1, -1 }), "deg must be non-negative");
            Raises<ValueError>(() => C.chebvander2d(a, b, new object[] { 1, long.MaxValue }), "Maximum allowed dimension exceeded");
            Raises<ValueError>(() => C.chebvander2d(a, b, new object[] { 1L << 61, 1.5 }), "array is too big;");   // dim 0's np.empty first
            Raises<TypeError>(() => C.chebvander2d(a, b, new object[] { 1.5, 1L << 61 }), "deg must be an integer, received 1.5");

            // Points: stacked, so a shape mismatch is np.array's ragged error — after len(deg) and the count, before the degrees.
            string ragged = "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. " +
                            "The detected shape was (2,) + inhomogeneous part.";
            Raises<ValueError>(() => C.chebvander2d(new[] { 1, 2 }, new[] { 3, 4, 5 }, new[] { 1, -1 }), ragged);
            Raises<ValueError>(() => C.chebvander2d(0.5, new[] { 1.0, 2.0 }, new[] { 1, 1 }), ragged);
            Raises<TypeError>(() => C.chebvander2d(new[] { 1, 2 }, new[] { 3 }, 2), "object of type 'int' has no len()");
            Raises<ValueError>(() => C.chebvander2d(new[] { 1, 2 }, new[] { 3 }, new[] { 1, 1, 1 }), "Expected 2 dimensions of degrees, got 3");
            FluentActions.Invoking(() => C.chebvander2d("ab", "cd", new[] { 1, 1 })).Should().Throw<NotSupportedException>();

            // No points: the degrees are still checked, then NumPy's reshape of the empty product fails.
            Raises<ValueError>(() => C.chebvander2d(np.zeros(0), np.zeros(0), new[] { 1, -1 }), "deg must be non-negative");
            Raises<IncorrectShapeException>(() => C.chebvander2d(np.zeros(0), np.zeros(0), new[] { 1, 2 }),
                "cannot reshape array of size 0 into shape (0,newaxis)");
            Raises<IncorrectShapeException>(() => C.chebvander3d(np.zeros(new Shape(3, 0, 2)), np.zeros(new Shape(3, 0, 2)), np.zeros(new Shape(3, 0, 2)),
                new[] { 1, 2, 1 }), "cannot reshape array of size 0 into shape (3,0,2,newaxis)");
        }

        /// <summary>
        ///     The dtype rules: <c>(x + 0.0).dtype</c> — bool / every integer / char are float64, float16 / float32 / complex128
        ///     kept; the 2-D / 3-D stack's promotion (array coercion's strong rule: float32 with int32 is float64, float16 with
        ///     int8 stays float16, a Python float is float64 there). char values equal the same values as uint16.
        /// </summary>
        [TestMethod]
        public void Vander_Dtypes_FollowNumPy()
        {
            var H = np.polynomial.hermite;
            foreach (var t in new[] { NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
                                      NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char })
                H.hermvander(np.arange(3).astype(t), 2).dtype.name.Should().Be("float64", t.ToString());
            H.hermvander(np.arange(3).astype(np.float16), 2).dtype.name.Should().Be("float16");
            H.hermvander(np.arange(3).astype(np.float32), 2).dtype.name.Should().Be("float32");
            H.hermvander(np.arange(3).astype(np.complex128), 2).dtype.name.Should().Be("complex128");
            Hex(H.hermvander(np.array(new[] { 'a', 'b', '\u0007' }), 4))
                .Should().Be(Hex(H.hermvander(np.array(new ushort[] { 'a', 'b', 7 }), 4)), "char computes exactly as uint16");

            H.hermvander2d(np.arange(3).astype(np.float32), np.arange(3).astype(np.int32), new[] { 1, 1 }).dtype.name.Should().Be("float64");
            H.hermvander2d(np.arange(3).astype(np.float16), np.arange(3).astype(np.int8), new[] { 1, 1 }).dtype.name.Should().Be("float16");
            H.hermvander2d(np.arange(3).astype(np.float16), np.arange(3).astype(np.int16), new[] { 1, 1 }).dtype.name.Should().Be("float32");
            H.hermvander2d(np.array((Half)0.5), 0.5, new[] { 1, 1 }).dtype.name.Should().Be("float64");
            H.hermvander2d((Half)0.5, (Half)0.25, new[] { 1, 1 }).dtype.name.Should().Be("float16");
            H.hermvander2d(np.arange(3).astype(np.int8), np.arange(3).astype(np.int8), new[] { 1, 1 }).dtype.name.Should().Be("float64");
            H.hermvander3d(1, 2, new Complex(0, 1), new[] { 1, 1, 1 }).dtype.name.Should().Be("complex128");
        }

        /// <summary>
        ///     NumSharp's decimal (no NumPy analog) runs the same statements in decimal arithmetic: a Legendre matrix checked
        ///     against the recurrence written out here, and a 2-D product.
        /// </summary>
        [TestMethod]
        public void Vander_Decimal_RunsTheRecurrenceInDecimal()
        {
            var x = np.array(new[] { 0.5m, -1.25m, 2m });
            var v = np.polynomial.legendre.legvander(x, 4);
            v.dtype.name.Should().Be(np.@decimal.name);
            var got = v.ToArray<decimal>();
            decimal[] xs = { 0.5m, -1.25m, 2m };
            for (int p = 0; p < 3; p++)
            {
                var row = new decimal[5];
                row[0] = xs[p] * 0m + 1m;
                row[1] = xs[p];
                for (int i = 2; i <= 4; i++)
                    row[i] = (row[i - 1] * xs[p] * (2 * i - 1) - row[i - 2] * (i - 1)) / i;
                for (int i = 0; i <= 4; i++)
                    got[p * 5 + i].Should().Be(row[i], $"point {p} degree {i}");
            }
            var v2 = np.polynomial.polynomial.polyvander2d(x, x, new[] { 1, 2 });
            v2.dtype.name.Should().Be(np.@decimal.name);
            v2.ToArray<decimal>()[5].Should().Be(0.5m * 0.25m);   // point 0, column (1, 2): x^1 * y^2
        }

        /// <summary>
        ///     The result is NumPy's VIEW: <c>np.moveaxis</c> of a fresh <c>(deg + 1,) + x.shape</c> buffer (1-D), the reshape of the
        ///     outer product (2-D / 3-D) — OWNDATA false, F-contiguous for a 1-D x, C- and F-contiguous for one point or degree 0
        ///     or no points, neither for an N-D x.
        /// </summary>
        [TestMethod]
        public void Vander_ResultIsNumPysView_Flags()
        {
            var E = np.polynomial.hermite_e;
            void Flags(NDArray r, bool c, bool f, string because)
            {
                r.flags.c_contiguous.Should().Be(c, because);
                r.flags.f_contiguous.Should().Be(f, because);
                r.flags.owndata.Should().BeFalse(because);
            }
            Flags(E.hermevander(np.arange(3.0), 2), false, true, "1-D x");
            Flags(E.hermevander(2.0, 2), true, true, "scalar x");
            Flags(E.hermevander(np.zeros(new Shape(2, 3)), 2), false, false, "2-D x");
            Flags(E.hermevander(np.zeros(0), 2), true, true, "no points");
            Flags(E.hermevander(np.arange(4.0), 0), true, true, "degree 0");
            Flags(E.hermevander2d(np.arange(4.0), np.arange(4.0), new[] { 2, 3 }), false, true, "2-D, 1-D points");
            Flags(E.hermevander2d(0.5, 0.25, new[] { 2, 1 }), true, true, "2-D, one point");
            Flags(E.hermevander3d(np.zeros(new Shape(2, 3)), np.zeros(new Shape(2, 3)), np.zeros(new Shape(2, 3)), new[] { 1, 1, 2 }), false, false, "3-D, 2-D points");
            // Writes go through to the view's buffer and never to x.
            var x = np.arange(3.0);
            var v = E.hermevander(x, 1);
            v[0, 1] = 99.0;
            v.GetDouble(0, 1).Should().Be(99.0);
            x.GetDouble(0).Should().Be(0.0);
        }

        /// <summary>
        ///     The <c>int</c> and <c>object</c> overloads are one function: every integer spelling of a degree (C# int / long,
        ///     BigInteger, a 0-d array) gives the same bytes, and x in any layout (strided, reversed, F-order, transposed —
        ///     read in place or copied) gives the bytes of its contiguous copy.
        /// </summary>
        [TestMethod]
        public void Vander_OverloadsAndLayouts_AgreeBitForBit()
        {
            foreach (var b in Bases)
            {
                var x = np.array(new[] { 0.3, -1.7, 2.9, 0.0, -0.0, 1e-3, 5.5, -2.25, 0.75 });
                string expect = Hex(b.VanderInt(x, 6));
                Hex(b.Vander(x, 6L)).Should().Be(expect, b.Name);
                Hex(b.Vander(x, new BigInteger(6))).Should().Be(expect, b.Name);
                Hex(b.Vander(x, np.array(6))).Should().Be(expect, b.Name);

                var wide = np.zeros(18);
                wide["::2"] = x;
                Hex(b.VanderInt(wide["::2"], 6)).Should().Be(expect, $"{b.Name} strided");
                Hex(b.VanderInt(x["::-1"]["::-1"], 6)).Should().Be(expect, $"{b.Name} double reverse");
                var m = x.reshape(3, 3);
                Hex(b.VanderInt(m.T, 5)).Should().Be(Hex(b.VanderInt(m.T.copy(), 5)), $"{b.Name} transposed");
                Hex(b.VanderInt(np.asfortranarray(m), 5)).Should().Be(Hex(b.VanderInt(m, 5)), $"{b.Name} F-order");
                Hex(b.Vander2d(m.T, m, new[] { 2, 3 })).Should().Be(Hex(b.Vander2d(m.T.copy(), m.copy(), new[] { 2, 3 })), $"{b.Name} 2-D layouts");
            }
        }

        // =========================================================================================================
        //  The kernel
        // =========================================================================================================

        /// <summary>
        ///     The block loop: forcing blocks of 1, 3, 16 and 17 points (the test hook) gives the bytes of the cache-sized default —
        ///     every block boundary, the partial last block and the per-block scratch reuse, for the 1-D form and the fused 2-D /
        ///     3-D outer product (whose default budget needs megabytes of output to span two blocks), every dtype family.
        /// </summary>
        [TestMethod]
        public void Kernels_ForcedSmallBlocks_AgreeWithTheDefault()
        {
            np.random.seed(11);
            var xs = new[]
            {
                np.random.random(new Shape(53)) * 3 - 1.5,
                (np.random.random(new Shape(53)) * 3 - 1.5).astype(np.float32),
                (np.random.random(new Shape(53)) * 3 - 1.5).astype(np.float16),
                np.random.random(new Shape(53)) * 2 - 1 + np.random.random(new Shape(53)).astype(np.complex128) * new Complex(0, 1),
                (np.random.random(new Shape(53)) * 20 - 10).astype(np.int32),
            };
            foreach (var b in Bases)
            {
                foreach (var x in xs)
                {
                    var y = x["::-1"].copy();
                    string v1 = Hex(b.VanderInt(x, 7)), v2 = Hex(b.Vander2d(x, y, new[] { 4, 3 })), v3 = Hex(b.Vander3d(x, y, x, new[] { 2, 3, 2 }));
                    foreach (long blk in new long[] { 1, 3, 16, 17 })
                    {
                        NDPolyVander.BlockOverride = blk;
                        try
                        {
                            Hex(b.VanderInt(x, 7)).Should().Be(v1, $"{b.Name} {x.dtype.name} 1-D block {blk}");
                            Hex(b.Vander2d(x, y, new[] { 4, 3 })).Should().Be(v2, $"{b.Name} {x.dtype.name} 2-D block {blk}");
                            Hex(b.Vander3d(x, y, x, new[] { 2, 3, 2 })).Should().Be(v3, $"{b.Name} {x.dtype.name} 3-D block {blk}");
                        }
                        finally
                        {
                            NDPolyVander.BlockOverride = 0;
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     The streamed product (non-temporal output stores, <see cref="PolyVanderKey.NonTemporal"/>) writes the normal
        ///     stores' bytes: forced on and off through the test hook, every basis and compute dtype (float64 / float32 / float16 /
        ///     complex128, int points as float64, and decimal — which has no such store and must keep its normal one), point
        ///     counts 1 … 1001 so every row's scalar head meets every alignment (a row starts at <c>r * npts * itemsize</c>, only
        ///     element-aligned), plus 3-point blocks that start every block mid-vector — 2-D and 3-D (whose scratch product row
        ///     keeps normal stores).
        /// </summary>
        [TestMethod]
        public void Kernels_NonTemporalProduct_WritesTheNormalStoresBytes()
        {
            np.random.seed(12);
            foreach (int n in new[] { 1, 2, 3, 5, 7, 9, 17, 33, 1001 })
            {
                var xs = new[]
                {
                    np.random.random(new Shape(n)) * 3 - 1.5,
                    (np.random.random(new Shape(n)) * 3 - 1.5).astype(np.float32),
                    (np.random.random(new Shape(n)) * 3 - 1.5).astype(np.float16),
                    np.random.random(new Shape(n)) * 2 - 1 + np.random.random(new Shape(n)).astype(np.complex128) * new Complex(0, 1),
                    (np.random.random(new Shape(n)) * 20 - 10).astype(np.int32),
                    (np.random.random(new Shape(n)) * 3 - 1.5).astype(np.@decimal),
                };
                foreach (var b in Bases)
                {
                    foreach (var x in xs)
                    {
                        var y = x["::-1"].copy();
                        foreach (long blk in new long[] { 0, 3 })
                        {
                            string[] got = new string[2];
                            for (int mode = 0; mode < 2; mode++)
                            {
                                NDPolyVander.NonTemporalOverride = mode == 0 ? -1 : 1;
                                NDPolyVander.BlockOverride = blk;
                                try
                                {
                                    got[mode] = Hex(b.Vander2d(x, y, new[] { 3, 4 })) + "|" + Hex(b.Vander3d(x, y, x, new[] { 2, 1, 3 }));
                                }
                                finally
                                {
                                    NDPolyVander.NonTemporalOverride = 0;
                                    NDPolyVander.BlockOverride = 0;
                                }
                            }
                            got[1].Should().Be(got[0], $"{b.Name} {x.dtype.name} n={n} block {blk}");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     A result past <c>NDPolyVander.NonTemporalMinBytes</c> (a 38.7 MB <c>(40000, 121)</c> float64 matrix and a 33.6 MB
        ///     3-D one) takes the streamed product on its own — and writes exactly what the normal stores write (compared by
        ///     SHA-256: the matrices are too large to dump). The oracle's matrices are all small, so this is the automatic path's gate.
        /// </summary>
        [TestMethod]
        public void Kernels_LargeResult_StreamedProduct_SameBytesAsNormalStores()
        {
            static string Sha(NDArray a)
            {
                using var c = np.ascontiguousarray(a);
                unsafe
                {
                    var span = new ReadOnlySpan<byte>((byte*)c.Storage.Address + c.Shape.offset * c.dtypesize, checked((int)(c.size * c.dtypesize)));
                    return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(span));
                }
            }
            np.random.seed(13);
            var x = np.random.random(new Shape(40_000)) * 2 - 1;
            var y = np.random.random(new Shape(40_000)) * 2 - 1;
            var z = np.random.random(new Shape(40_000)) * 2 - 1;
            var cheb = Bases[1];
            (40_000L * 121 * 8).Should().BeGreaterThanOrEqualTo(NDPolyVander.NonTemporalMinBytes, "the 2-D result must cross the threshold");
            (40_000L * 105 * 8).Should().BeGreaterThanOrEqualTo(NDPolyVander.NonTemporalMinBytes, "the 3-D result must cross the threshold");

            string auto2, auto3, normal2, normal3;
            using (var r = cheb.Vander2d(x, y, new[] { 10, 10 })) auto2 = Sha(r);
            using (var r = cheb.Vander3d(x, y, z, new[] { 4, 2, 6 })) auto3 = Sha(r);
            NDPolyVander.NonTemporalOverride = -1;
            try
            {
                using (var r = cheb.Vander2d(x, y, new[] { 10, 10 })) normal2 = Sha(r);
                using (var r = cheb.Vander3d(x, y, z, new[] { 4, 2, 6 })) normal3 = Sha(r);
            }
            finally
            {
                NDPolyVander.NonTemporalOverride = 0;
            }
            auto2.Should().Be(normal2, "2-D");
            auto3.Should().Be(normal3, "3-D");
        }

        /// <summary>
        ///     Every compute dtype and every source dtype the loads convert has vector lanes on this host: compiling the kernels of
        ///     every dtype (1-D) and of mixed-dtype points (2-D) never falls back to scalar-only code.
        /// </summary>
        [TestMethod]
        public void Kernels_EveryDtypeHasVectorLanes_NoScalarFallback()
        {
            long before = DirectILKernelGenerator.PolyVanderVectorFallbacks;
            foreach (var b in Bases)
            {
                foreach (var t in new[] { NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
                                          NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
                                          NPTypeCode.Double, NPTypeCode.Complex, NPTypeCode.Decimal })
                    b.VanderInt(np.arange(19).astype(t), 3);
                b.Vander2d(np.arange(19).astype(np.float16), np.arange(19).astype(np.int8), new[] { 2, 2 });
                b.Vander2d(np.arange(19).astype(np.float32), np.arange(19).astype(np.int16), new[] { 2, 2 });
                b.Vander3d(np.arange(19).astype(np.complex128), np.arange(19).astype(np.float64), np.arange(19).astype(np.int32), new[] { 1, 2, 1 });
            }
            DirectILKernelGenerator.PolyVanderVectorFallbacks.Should().Be(before);
        }
    }
}
