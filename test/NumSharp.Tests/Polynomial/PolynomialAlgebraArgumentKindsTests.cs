using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The wholeness pass over the <c>numpy.polynomial</c> series algebra (plan U2): the C# spellings of its
    ///     arguments that the byte oracle (<c>polyalgebra.jsonl</c>, which replays only <c>object[]</c>,
    ///     <c>ValueTuple</c>, Python scalars, np.float16 and NDArrays) cannot express — <c>List&lt;T&gt;</c>, LINQ,
    ///     <c>NDArray[]</c>, jagged and multi-dimensional arrays; pow's integer power and limit (typed since API-parity
    ///     batch 11); the object / str series refusal deferred to where NumPy computes with Python objects; and the
    ///     complex dotfunc behind <c>np.convolve</c> (zdotu's C99 result construction, CDOUBLE_dot's plain loop for a
    ///     one-element operand). Every expected value is NumPy 2.4.2's, the Python spelling beside it.
    /// </summary>
    [TestClass]
    public class PolynomialAlgebraArgumentKindsTests
    {
        private static readonly PowerSeriesModule P = np.polynomial.polynomial;
        private static readonly ChebyshevModule C = np.polynomial.chebyshev;

        /// <summary>Asserts dtype, shape and every byte (logical C order) against a NumPy hex dump — bit parity.</summary>
        /// <param name="actual">The result.</param>
        /// <param name="dtype">NumPy's dtype name.</param>
        /// <param name="shape">NumPy's shape.</param>
        /// <param name="hex">NumPy's <c>tobytes().hex()</c> of the C-order copy.</param>
        /// <param name="because">What the assertion checks (reported on failure).</param>
        private static void AssertBytes(NDArray actual, string dtype, long[] shape, string hex, string because)
        {
            actual.dtype.name.Should().Be(dtype, because);
            actual.shape.Should().Equal(shape, because);
            Hex(actual).Should().Be(hex, because);
        }

        /// <summary>The logical-C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array (any layout).</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a)
        {
            // np.ascontiguousarray hands a C-contiguous operand back itself: only a copy it made is this helper's to
            // dispose (disposing a would free the caller's array under it).
            var c = np.ascontiguousarray(a);
            using var cCopy = ReferenceEquals(c, a) ? null : c;
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        /// <summary>Asserts that <paramref name="call"/> raises <typeparamref name="TException"/> with NumPy's exact text.</summary>
        /// <typeparam name="TException">The exception type NumPy's maps to.</typeparam>
        /// <param name="call">The call.</param>
        /// <param name="message">NumPy's message.</param>
        /// <param name="because">The Python spelling (reported on failure).</param>
        private static void Raises<TException>(Action call, string message, string because) where TException : Exception
            => call.Should().Throw<TException>(because).Which.Message.Should().Be(message, because);

        /// <summary>NumPy's <c>chebpow([1.0, 2.0], k)</c> for k = 0, 1, 2, 3.</summary>
        private static readonly string[] ChebPowHex =
        {
            "000000000000f03f",
            "000000000000f03f0000000000000040",
            "000000000000084000000000000010400000000000000040",
            "0000000000001c40000000000000284000000000000018400000000000000040",
        };

        /// <summary>The series <c>[1.0, 2.0]</c> the pow tests raise.</summary>
        private static NDArray C12() => np.array(new[] { 1.0, 2.0 });

        /// <summary>Asserts <paramref name="r"/> is NumPy's <c>chebpow([1.0, 2.0], k)</c>.</summary>
        /// <param name="r">The result.</param>
        /// <param name="k">The power it must equal.</param>
        /// <param name="because">The spelling (reported on failure).</param>
        private static void IsChebPow(NDArray r, int k, string because)
            => AssertBytes(r, "float64", new long[] { k + 1 }, ChebPowHex[k], because);

        /// <summary>
        ///     Series spellings NumPy reads through <c>np.array</c> (as_series): a <c>List&lt;T&gt;</c> or LINQ sequence is a
        ///     Python list of its items (a C# float item a Python float, a Half item np.float16), an <c>NDArray[]</c> a list
        ///     of ndarrays (promoted strongly: 0-d float32 and int64 give float64), a jagged or 2-D array an N-D array
        ///     ("not 1-d"), a ragged one np.array's inhomogeneous error, a <c>ulong[]</c> a uint64 ndarray.
        /// </summary>
        [TestMethod]
        public void SeriesSpellings_AreNumPysArrayLikes()
        {
            var c2 = new[] { 0.5, -1.0, 2.0 };
            // polymul([1.5, -2.0, 0.25] (Python floats), [0.5, -1., 2.])
            AssertBytes(P.polymul(new List<float> { 1.5f, -2f, 0.25f }, c2), "float64", new long[] { 5 },
                        "000000000000e83f00000000000004c0000000000080144000000000000011c0000000000000e03f", "List<float>");
            // polymul([1.5, -2.0], ...) spelled as a lazy LINQ sequence
            AssertBytes(P.polymul(new object[] { 1.5, -2.0 }.Select(x => x), c2), "float64", new long[] { 4 },
                        "000000000000e83f00000000000004c0000000000000144000000000000010c0", "LINQ");
            // polymul([np.float32(1.5), np.int64(2)] as 0-d arrays, ...)
            AssertBytes(P.polymul(new[] { NDArray.Scalar(1.5f), NDArray.Scalar(2L) }, c2), "float64", new long[] { 4 },
                        "000000000000e83f000000000000e0bf000000000000f03f0000000000001040", "NDArray[] of 0-d arrays");
            // chebmul([0.5, -1., 2.], [1+2j, -2+0j])
            AssertBytes(C.chebmul(c2, new List<Complex> { new(1, 2), new(-2, 0) }), "complex128", new long[] { 4 },
                        "000000000000f83f000000000000f03f00000000000010c000000000000000c00000000000000840000000000000104000000000000000c00000000000000000",
                        "List<Complex>");
            // hermpow([np.float16(1.5), 2], 2)
            AssertBytes(np.polynomial.hermite.hermpow(new object[] { (Half)1.5, 2 }, 2), "float64", new long[] { 3 },
                        "000000000080244000000000000018400000000000001040", "a Half item beside a Python int");
            // polyfromroots([np.float16(1.5), np.float16(-2.)]) — List<Half> items are np.float16 scalars
            AssertBytes(P.polyfromroots(new List<Half> { (Half)1.5, (Half)(-2) }), "float64", new long[] { 3 },
                        "00000000000008c0000000000000e03f000000000000f03f", "List<Half>");
            // lagfromroots(np.array([3, 2, 2**64-1], np.uint64))
            AssertBytes(np.polynomial.laguerre.lagfromroots(new ulong[] { 3, 2, ulong.MaxValue }), "float64", new long[] { 4 },
                        "00000000000008c4000000000000f0c300000000000000c400000000000018c0", "ulong[]");
            // polypow(np.float16(1.5), 2); hermemulx(np.array([1.5, -2., 0.25], np.float16))
            AssertBytes(P.polypow((Half)1.5, 2), "float16", new long[] { 1 }, "8040", "a Half series");
            AssertBytes(np.polynomial.hermite_e.hermemulx(new Half[] { (Half)1.5, (Half)(-2), (Half)0.25 }), "float16", new long[] { 4 },
                        "00c0004000c00034", "Half[]");

            Raises<ValueError>(() => C.chebpow(new double[,] { { 1.5, -2, 0.25 } }, 2), "Coefficient array is not 1-d", "a 1x3 double[,]");
            Raises<ValueError>(() => P.polyfromroots(new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } }), "Coefficient array is not 1-d",
                               "a jagged double[][]");
            Raises<ValueError>(() => C.chebmulx(new[] { new[] { 1.0, 2.0 }, new[] { 3.0 } }),
                               "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 " +
                               "dimensions. The detected shape was (2,) + inhomogeneous part.", "a ragged jagged array");
            Raises<ValueError>(() => C.cheb2poly(new List<bool> { true, false }), "Coefficient arrays have no common type", "List<bool>");
            Raises<ValueError>(() => np.polynomial.hermite_e.poly2herme(Enumerable.Empty<object>()), "Coefficient array is empty",
                               "an empty LINQ sequence");
            Raises<ZeroDivisionError>(() => P.polydiv(c2, 0), "", "polydiv(c, 0)");
        }

        /// <summary>
        ///     The power and the limit of <c>{p}pow</c> are NumPy's integers, typed: <c>(c, int pow, int? maxpower)</c> is
        ///     every basis' one signature since API-parity batch 11 (the owner's decision 2, typed parameters only). NumPy's
        ///     statements over them still run in NumPy's order - <c>power != pow or power &lt; 0</c>, then
        ///     <c>power &gt; maxpower</c> for a limit that is not None, then the product - while NumPy's <c>int()</c> of a
        ///     bool, float, str, decimal, BigInteger or array power and its comparisons with a float, complex, str or array
        ///     limit have no C# call: the compiler refuses those arguments.
        /// </summary>
        [TestMethod]
        public void Pow_PowerAndLimit_AreNumPysIntegers()
        {
            var c = C12();
            for (int k = 0; k <= 3; k++)
                IsChebPow(C.chebpow(c, k), k, $"chebpow(c, {k})");
            const string TooLarge = "Power is too large";
            IsChebPow(C.chebpow(c, 2, 2), 2, "chebpow(c, 2, 2): the limit is inclusive");
            Raises<ValueError>(() => C.chebpow(c, 3, 2), TooLarge, "chebpow(c, 3, 2)");
            Raises<ValueError>(() => C.chebpow(c, 17), TooLarge, "chebpow(c, 17) at the default maxpower 16");
            Raises<ValueError>(() => C.chebpow(c, 0, -1), TooLarge, "chebpow(c, 0, -1): 0 > -1");
            C.chebpow(c, 20, null).size.Should().Be(21, "maxpower=None: no limit");
            Raises<ValueError>(() => C.chebpow(c, -1), "Power must be a non-negative integer.", "chebpow(c, -1)");
            AssertBytes(P.polypow(c, 3, 5), "float64", new long[] { 4 },
                        "000000000000f03f000000000000184000000000000028400000000000002040", "polypow(c, 3, 5)");
            // The limit decides before the object series' refusal (NumPy's object arithmetic) - exceeding raises first.
            Raises<ValueError>(() => P.polypow(new object[] { null }, 3, 2), TooLarge, "polypow([None], 3, 2)");
            ((Action)(() => P.polypow(new object[] { null }, 3, 5))).Should().Throw<NotSupportedException>(
                "polypow([None], 3, 5) passes the limit and reaches the object product");

            foreach (var (module, name) in new[]
                     {
                         (typeof(PowerSeriesModule), "polypow"), (typeof(ChebyshevModule), "chebpow"),
                         (typeof(LegendreModule), "legpow"), (typeof(LaguerreModule), "lagpow"),
                         (typeof(HermiteModule), "hermpow"), (typeof(HermiteEModule), "hermepow"),
                     })
            {
                var forms = module.GetMethods().Where(m => m.Name == name).ToArray();
                forms.Should().ContainSingle(name);
                forms[0].GetParameters().Select(p => p.ParameterType).Should()
                    .Equal(new[] { typeof(object), typeof(int), typeof(int?) }, name);
            }
        }

        /// <summary>
        ///     A series NumPy makes an object array (None, a non-numeric item, an oversized int) or a str array is checked
        ///     like any other before anything computes: as_series' size and dims for every argument, np.array's raggedness,
        ///     the str array's missing common type, div's zero divisor and pow's checks — each NumPy's error, in NumPy's
        ///     order; only where NumPy then computes with Python objects does NumSharp (no object dtype) refuse. The
        ///     evaluation (U3) and calculus (U4) families compute at once and keep refusing on conversion.
        /// </summary>
        [TestMethod]
        public void ObjectAndStrSeries_RefusedWhereNumPyComputes()
        {
            Raises<ValueError>(() => P.polymul(null, np.zeros(0)), "Coefficient array is empty", "polymul(None, [])");
            Raises<ValueError>(() => P.polymul(new object[] { 1.0, "a" }, new[] { 1.0 }), "Coefficient arrays have no common type",
                               "polymul([1.0, 'a'], [1.0])");
            Raises<ValueError>(() => P.polymul(null, new object[] { new object[] { 1, 2 }, new object[] { 3 } }),
                               "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 " +
                               "dimensions. The detected shape was (2,) + inhomogeneous part.", "polymul(None, [[1, 2], [3]])");
            Raises<ZeroDivisionError>(() => P.polydiv(null, new[] { 0.0 }), "", "polydiv(None, [0.])");
            Raises<ZeroDivisionError>(() => P.polydiv(new object[] { null }, new[] { false }), "",
                                          "polydiv([None], [False]): the zero test runs on the object copies");
            ((Action)(() => P.polydiv(null, new[] { 1.0 }))).Should().Throw<NotSupportedException>("None / 1.0 is object arithmetic");
            Raises<ValueError>(() => P.polypow(null, -1), "Power must be a non-negative integer.", "polypow(None, -1)");
            ((Action)(() => P.polypow(null, 2))).Should().Throw<NotSupportedException>("NumPy multiplies the objects");
            ((Action)(() => P.polypow(null, 0))).Should().Throw<NotSupportedException>("NumPy returns an object array [1]");
            Raises<ValueError>(() => C.chebpow(new object[] { 1.0, null }, 17), "Power is too large", "chebpow([1.0, None], 17)");
            Raises<ValueError>(() => C.chebmulx(new object[] { new object[] { null, 1.0 } }), "Coefficient array is not 1-d",
                               "chebmulx([[None, 1.0]])");
            Raises<ValueError>(() => P.polyadd(null, np.zeros(0)), "Coefficient array is empty", "polyadd(None, [])");
            Raises<ValueError>(() => np.polynomial.polyutils.as_series(new object[] { null, np.zeros(0) }), "Coefficient array is empty",
                               "as_series([None, []])");
            Raises<ValueError>(() => P.polyfromroots(new object[] { 1.0, "a" }), "Coefficient arrays have no common type",
                               "polyfromroots([1.0, 'a'])");

            ((Action)(() => C.chebval(1.0, new object[] { 1.0, null }))).Should().Throw<NotSupportedException>("U3 computes at once");
            ((Action)(() => C.chebder(new object[] { 1.0, null }))).Should().Throw<NotSupportedException>("U4 computes at once");
        }

        /// <summary>
        ///     NumPy's complex dotfunc behind np.convolve / np.correlate: <c>cblas_zdotu_sub</c> returns C99 complex
        ///     <c>re + im*_Complex_I</c>, so an infinite / NaN imaginary part turns the real part into NaN — but only when
        ///     both operands reach it with a positive stride. A ONE-element operand is passed through with its own stride
        ///     (it is C-contiguous whatever the stride), and np.convolve's kernel is <c>v[::-1]</c>: a fresh one-element
        ///     kernel arrives with stride -16 and takes CDOUBLE_dot's plain loop (no such NaN), a one-element view already
        ///     reversed arrives with +16 and takes zdotu.
        /// </summary>
        [TestMethod]
        public void ComplexDot_IsZdotuOrThePlainLoop()
        {
            var engine = BackendFactory.GetEngine();
            var saved = engine.Blas;
            engine.Blas = null;
            try
            {
                var inf = double.PositiveInfinity;
                var a = np.array(new[] { new Complex(1.5, -0.5), new Complex(0, inf), new Complex(-2, 1) });
                var v = np.array(new[] { new Complex(inf, 1), new Complex(2, 3) });
                var one = np.array(new[] { new Complex(inf, 1) });
                var rev = one["::-1"];
                // The operands hold infinities but no NaN, so every NaN below is GENERATED (inf*0, inf-inf). Two routes,
                // two rules: the zdotu positions (Z) end in NDDot's NaN model (D23), which writes x86's default NaN on EVERY
                // host, NumPy win-amd64's bits; the plain loop (N) is plain double arithmetic, so it holds the CPU's own
                // default NaN (arm64's is positive, see HostCpu).
                const string Z = "000000000000f8ff";
                string N = HostCpu.DefaultNaN64Hex;
                AssertBytes(np.convolve(a, v), "complex128", new long[] { 4 },
                            $"{Z}000000000000f0ff{Z}000000000000f07f{Z}000000000000f07f0000000000001cc000000000000010c0",
                            "np.convolve(a, v): zdotu");
                AssertBytes(np.convolve(a, one), "complex128", new long[] { 3 },
                            $"000000000000f07f000000000000f0ff{N}000000000000f07f000000000000f0ff000000000000f07f",
                            "np.convolve(a, one): one[::-1] has stride -16, the plain loop");
                AssertBytes(np.convolve(a, rev), "complex128", new long[] { 3 },
                            $"{Z}000000000000f0ff{Z}000000000000f07f{Z}000000000000f07f",
                            "np.convolve(a, one[::-1]): reversed again, stride +16, zdotu");
                AssertBytes(np.convolve(one, one), "complex128", new long[] { 1 }, "000000000000f07f000000000000f07f",
                            "np.convolve(one, one): the kernel reversed, the plain loop");
                AssertBytes(np.correlate(a, one, "full"), "complex128", new long[] { 3 },
                            $"{Z}000000000000f0ff{Z}000000000000f07f{Z}000000000000f07f",
                            "np.correlate(a, one): conj(one) is a fresh array, zdotu");
                AssertBytes(np.correlate(one, a, "full"), "complex128", new long[] { 3 },
                            $"{Z}000000000000f0ff{Z}000000000000f0ff{Z}000000000000f07f",
                            "np.correlate(one, a): one keeps stride +16, zdotu");
                AssertBytes(np.correlate(rev, a, "full"), "complex128", new long[] { 3 },
                            $"000000000000f0ff000000000000f0ff{N}000000000000f0ff000000000000f07f000000000000f07f",
                            "np.correlate(one[::-1], a): stride -16, the plain loop");
            }
            finally
            {
                engine.Blas = saved;
            }
        }
    }
}
