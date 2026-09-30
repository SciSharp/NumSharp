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
    ///     <c>NDArray[]</c>, jagged and multi-dimensional arrays, decimal / char / BigInteger powers and limits, typed
    ///     C# arrays as powers and limits; the object / str series refusal deferred to where NumPy computes with Python
    ///     objects; and the complex dotfunc behind <c>np.convolve</c> (zdotu's C99 result construction, CDOUBLE_dot's
    ///     plain loop for a one-element operand). Every expected value is NumPy 2.4.2's, the Python spelling beside it.
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
            using var c = np.ascontiguousarray(a);
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
            Raises<DivideByZeroException>(() => P.polydiv(c2, 0), "", "polydiv(c, 0)");
        }

        /// <summary>
        ///     Any power argument through the object-typed overloads: <c>power = int(pow)</c> with Python's semantics — a
        ///     bool is 0 / 1, np.float16 / decimal truncate like a float, a str parses with int()'s grammar and then fails
        ///     <c>power != pow</c>, a 0-d NDArray converts its element — and NumPy's error for everything int() refuses.
        /// </summary>
        [TestMethod]
        public void Pow_AnyPowerArgument_IsPythonsInt()
        {
            var c = C12();
            IsChebPow(C.chebpow(c, true), 1, "chebpow(c, True)");
            IsChebPow(C.chebpow(c, false), 0, "chebpow(c, False)");
            IsChebPow(C.chebpow(c, (Half)2), 2, "chebpow(c, np.float16(2.0))");
            IsChebPow(C.chebpow(c, (object)'\u0002'), 2, "chebpow(c, np.uint16(2)) — char through the object overload");
            IsChebPow(C.chebpow(c, 2m), 2, "a decimal power truncates like a float");
            IsChebPow(C.chebpow(c, new BigInteger(3)), 3, "a BigInteger power is a Python int");
            IsChebPow(C.chebpow(c, NDArray.Scalar(3L)), 3, "chebpow(c, np.array(3))");
            IsChebPow(C.chebpow(c, NDArray.Scalar(true)), 1, "chebpow(c, np.array(True))");
            IsChebPow(C.chebpow(c, NDArray.Scalar((Half)2)), 2, "chebpow(c, np.array(2, np.float16))");

            const string NotInt = "Power must be a non-negative integer.";
            Raises<ValueError>(() => C.chebpow(c, 2.5m), NotInt, "a fractional decimal");
            Raises<ValueError>(() => C.chebpow(c, (Half)2.5), NotInt, "np.float16(2.5)");
            Raises<ValueError>(() => C.chebpow(c, NDArray.Scalar(2.5)), NotInt, "np.array(2.5)");
            Raises<ValueError>(() => C.chebpow(c, NDArray.Scalar((sbyte)-1)), NotInt, "np.array(-1, np.int8)");
            Raises<ValueError>(() => C.chebpow(c, "3"), NotInt, "'3' parses, but 3 != '3'");
            Raises<ValueError>(() => C.chebpow(c, " 3 "), NotInt, "' 3 ' parses too");
            Raises<ValueError>(() => C.chebpow(c, "x"), "invalid literal for int() with base 10: 'x'", "chebpow(c, 'x')");
            Raises<ValueError>(() => C.chebpow(c, ""), "invalid literal for int() with base 10: ''", "chebpow(c, '')");
            Raises<ValueError>(() => C.chebpow(c, Half.NaN), "cannot convert float NaN to integer", "np.float16('nan')");
            Raises<OverflowException>(() => C.chebpow(c, Half.PositiveInfinity), "cannot convert float infinity to integer",
                                      "np.float16('inf')");
            Raises<ValueError>(() => C.chebpow(c, BigInteger.Pow(2, 70)), "Power is too large", "chebpow(c, 2**70)");
            Raises<ValueError>(() => C.chebpow(c, NDArray.Scalar((byte)250)), "Power is too large", "np.array(250, np.uint8)");
            Raises<OverflowException>(() => P.polypow(c, BigInteger.Pow(2, 70)),
                                      "power 1180591620717411303424 is beyond int64: NumPy would run out of memory computing it",
                                      "polypow has no limit: NumPy would multiply until memory runs out");

            const string IntArg = "int() argument must be a string, a bytes-like object or a real number, not ";
            Raises<TypeError>(() => C.chebpow(c, null), IntArg + "'NoneType'", "chebpow(c, None)");
            Raises<TypeError>(() => C.chebpow(c, new object[] { 3 }), IntArg + "'list'", "chebpow(c, [3])");
            Raises<TypeError>(() => C.chebpow(c, new List<int> { 3 }), IntArg + "'list'", "a List<int> is a Python list");
            Raises<TypeError>(() => C.chebpow(c, ValueTuple.Create(3)), IntArg + "'tuple'", "chebpow(c, (3,))");
            Raises<TypeError>(() => C.chebpow(c, new Complex(2, 0)), IntArg + "'complex'", "chebpow(c, 2+0j)");
            Raises<TypeError>(() => C.chebpow(c, NDArray.Scalar(new Complex(2, 0))), IntArg + "'complex'", "np.array(2+0j)");
            const string ZeroDim = "only 0-dimensional arrays can be converted to Python scalars";
            Raises<TypeError>(() => C.chebpow(c, np.array(new[] { 3 })), ZeroDim, "np.array([3])");
            Raises<TypeError>(() => C.chebpow(c, new[] { 3 }), ZeroDim, "an int[] is an ndarray of one dim");
            Raises<TypeError>(() => C.chebpow(c, Array.Empty<double>()), ZeroDim, "an empty double[]");
            Raises<TypeError>(() => C.chebpow(c, np.array(new[,] { { 3 } })), ZeroDim, "np.array([[3]])");
        }

        /// <summary>
        ///     Any limit through the three-argument object overload: <c>power &gt; maxpower</c> as NumPy evaluates it —
        ///     exact for Python numbers, decimal, char and BigInteger; NaN never exceeded, -inf always; in float16 for
        ///     np.float16 (2049 rounds to 2048: not exceeded); an ndarray's comparison in its dtype, then tested for truth
        ///     (one element only); and the weak int's conversion errors (past 2**1024 into a float loop, past int64 into a
        ///     bool one) before that test.
        /// </summary>
        [TestMethod]
        public void Pow_AnyMaxpower_ComparesAsNumPy()
        {
            var c = C12();
            const string TooLarge = "Power is too large";
            IsChebPow(C.chebpow(c, 2, 2.5), 2, "chebpow(c, 2, 2.5)");
            Raises<ValueError>(() => C.chebpow(c, 3, 2.5), TooLarge, "chebpow(c, 3, 2.5)");
            C.chebpow(c, 20, double.NaN).size.Should().Be(21, "power > nan is False: no limit");
            Raises<ValueError>(() => C.chebpow(c, 0, double.NegativeInfinity), TooLarge, "0 > -inf");
            IsChebPow(C.chebpow(c, 2, 2.5m), 2, "a decimal limit compares exactly");
            Raises<ValueError>(() => C.chebpow(c, 3, 2.5m), TooLarge, "3 > 2.5 (decimal)");
            IsChebPow(C.chebpow(c, (object)3, (object)'\u0005'), 3, "chebpow(c, 3, np.uint16(5))");
            Raises<ValueError>(() => C.chebpow(c, (object)3, (object)'\u0002'), TooLarge, "chebpow(c, 3, np.uint16(2))");
            IsChebPow(C.chebpow(c, 3, new BigInteger(5)), 3, "a BigInteger limit");
            IsChebPow(C.chebpow(c, 3, new[] { 5.0 }), 3, "a one-element double[] compares, then is true or false");
            AssertBytes(P.polypow(c, 3, np.array(new[] { 5 })), "float64", new long[] { 4 },
                        "000000000000f03f000000000000184000000000000028400000000000002040", "polypow(c, 3, np.array([5]))");
            // 2049 rounds to float16 2048, so NumPy computes the power; 2050 is representable and exceeds it.
            AssertBytes(P.polypow(new[] { 1.5 }, 2049, (Half)2048), "float64", new long[] { 1 }, "000000000000f07f",
                        "polypow([1.5], 2049, np.float16(2048))");
            Raises<ValueError>(() => P.polypow(new[] { 1.5 }, 2050, (Half)2048), TooLarge, "polypow([1.5], 2050, np.float16(2048))");

            const string Cmp = "'>' not supported between instances of 'int' and ";
            Raises<TypeError>(() => C.chebpow(c, 2, new Complex(5, 0)), Cmp + "'complex'", "a complex limit");
            Raises<TypeError>(() => C.chebpow(c, 2, "5"), Cmp + "'str'", "a str limit");
            Raises<TypeError>(() => C.chebpow(c, 2, new object[] { 5 }), Cmp + "'list'", "a list limit");
            Raises<TypeError>(() => C.chebpow(c, 2, ValueTuple.Create(5)), Cmp + "'tuple'", "a tuple limit");
            Raises<ValueError>(() => P.polypow(c, 3, np.array(new[] { 1, 5 })),
                               "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()",
                               "a two-element limit array");
            Raises<ValueError>(() => P.polypow(c, 3, np.zeros(0)),
                               "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.",
                               "an empty limit array");

            // The comparison dtype decides what passes; an OBJECT series then shows which way it went without computing:
            // passing reaches the object refusal (NumPy: its object arithmetic's TypeError), exceeding raises first.
            var obj = new object[] { null };
            var p = BigInteger.Pow(2, 60) + BigInteger.Pow(2, 36) + 1;
            ((Action)(() => P.polypow(obj, p, NDArray.Scalar((float)System.Math.Pow(2, 60))))).Should().Throw<NotSupportedException>(
                "the power rounds to float32 2**60 (through a double first): not exceeded");
            Raises<ValueError>(() => P.polypow(obj, p, System.Math.Pow(2, 60)), TooLarge, "a Python float compares exactly");
            Raises<OverflowException>(() => P.polypow(obj, BigInteger.Pow(2, 63), NDArray.Scalar(true)), "int too big to convert",
                                      "a bool limit compares in int64");
            Raises<OverflowException>(() => P.polypow(obj, BigInteger.Pow(2, 1100), NDArray.Scalar(1.0)),
                                      "int too large to convert to float", "a float64 limit converts the power to a double");
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
            Raises<DivideByZeroException>(() => P.polydiv(null, new[] { 0.0 }), "", "polydiv(None, [0.])");
            Raises<DivideByZeroException>(() => P.polydiv(new object[] { null }, new[] { false }), "",
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
                AssertBytes(np.convolve(a, v), "complex128", new long[] { 4 },
                            "000000000000f8ff000000000000f0ff000000000000f8ff000000000000f07f000000000000f8ff000000000000f07f0000000000001cc000000000000010c0",
                            "np.convolve(a, v): zdotu");
                AssertBytes(np.convolve(a, one), "complex128", new long[] { 3 },
                            "000000000000f07f000000000000f0ff000000000000f8ff000000000000f07f000000000000f0ff000000000000f07f",
                            "np.convolve(a, one): one[::-1] has stride -16, the plain loop");
                AssertBytes(np.convolve(a, rev), "complex128", new long[] { 3 },
                            "000000000000f8ff000000000000f0ff000000000000f8ff000000000000f07f000000000000f8ff000000000000f07f",
                            "np.convolve(a, one[::-1]): reversed again, stride +16, zdotu");
                AssertBytes(np.convolve(one, one), "complex128", new long[] { 1 }, "000000000000f07f000000000000f07f",
                            "np.convolve(one, one): the kernel reversed, the plain loop");
                AssertBytes(np.correlate(a, one, "full"), "complex128", new long[] { 3 },
                            "000000000000f8ff000000000000f0ff000000000000f8ff000000000000f07f000000000000f8ff000000000000f07f",
                            "np.correlate(a, one): conj(one) is a fresh array, zdotu");
                AssertBytes(np.correlate(one, a, "full"), "complex128", new long[] { 3 },
                            "000000000000f8ff000000000000f0ff000000000000f8ff000000000000f0ff000000000000f8ff000000000000f07f",
                            "np.correlate(one, a): one keeps stride +16, zdotu");
                AssertBytes(np.correlate(rev, a, "full"), "complex128", new long[] { 3 },
                            "000000000000f0ff000000000000f0ff000000000000f8ff000000000000f0ff000000000000f07f000000000000f07f",
                            "np.correlate(one[::-1], a): stride -16, the plain loop");
            }
            finally
            {
                engine.Blas = saved;
            }
        }
    }
}
