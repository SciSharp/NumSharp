using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     Interpolation at the Chebyshev points of <c>numpy.polynomial</c> (plan
    ///     <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.5, unit U9): <c>chebinterpolate</c>. NumPy 2.4.2's own
    ///     <c>TestInterpolate</c> ported; NumPy's bytes for small degrees; NumPy's outcome for every argument kind with a C#
    ///     spelling (the probe script <c>ci_pins.py</c> is quoted next to <see cref="Pins"/>); Python's <c>*args</c> unpacking;
    ///     and the composition against NumPy's statements written out by hand.
    /// </summary>
    /// <remarks>
    ///     The values come out of <c>np.dot</c>: with the OpenBLAS backend (the scipy-openblas NumPy itself calls, at one
    ///     thread) they are NumPy's bytes, which <see cref="Pins_WithTheBackend_AreNumPysBytes"/> asserts; without it the
    ///     managed product is held to a few units of rounding of the largest coefficient. Byte parity over 1,030 cases is the
    ///     host-pinned <c>PolyinterpParity</c> oracle tier. <c>[DoNotParallelize]</c>: the tests bind and unbind the
    ///     process-wide OpenBLAS backend (<c>TensorEngine.Blas</c>).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class PolynomialInterpolateTests
    {
        /// <summary>The functions of <see cref="Pins"/>, by name: the twins of the probe's Python callables.</summary>
        private static readonly Dictionary<string, Func<NDArray, object[], object>> Functions = new()
        {
            ["tanh_half"] = (x, _) => np.tanh(x) + 0.5,
            ["cubic"] = (x, _) => x * x * x - 2.0 * x + 1.0,
            ["expi"] = (x, _) => np.exp(np.multiply(NDArray.Scalar(new Complex(0, 1)), x)),
            ["stack2"] = (x, _) => np.stack(new[] { np.sin(x), np.cos(x) }, axis: 1),
            ["pyfloat"] = (x, _) => 2.0,
            ["i64"] = (x, _) => (x * 10).astype(np.int64),
        };

        // NumPy 2.4.2's bytes (scratchpad ci_pins.py, OPENBLAS_NUM_THREADS=1): for each function and degree,
        // r = C.chebinterpolate(f, d); r.dtype, r.shape and (r.view(float64) for complex) r.tobytes().hex().
        /// <summary>NumPy's result of each (function, degree): dtype, shape and the float64 bytes.</summary>
        private static readonly (string Fn, int Deg, string Dtype, long[] Shape, string Hex)[] Pins =
        {
            ("tanh_half", 0, "float64", new long[] { 1 }, "000000000000e03f"),
            ("tanh_half", 1, "float64", new long[] { 2 }, "000000000000e03f3f97fdc4c78deb3f"),
            ("tanh_half", 2, "float64", new long[] { 3 }, "000000000000e03f9f075a965bd7e93f75e2ebd5c1d69dbc"),
            ("tanh_half", 5, "float64", new long[] { 6 }, "000000000000e03f7b9ee39144f9e93f55555555555595bcd0619c535bcaabbf000000000000b0bc40a9e60f4111743f"),
            ("tanh_half", 12, "float64", new long[] { 13 }, "000000000000e03f9a2353493ff9e93f8aad9173ecfca2bcbc75955a1bc6abbfa76ee9b831b7aabc8452062b3b80723f96d18372eacca7bc3c441f7a651039bff9406913a391b7bc3f0326c95302013fd9a6b4d145bf90bc91fe9bd758edc6be59a15df6efe5bb3c"),
            ("cubic", 0, "float64", new long[] { 1 }, "000000000000f03f"),
            ("cubic", 1, "float64", new long[] { 2 }, "000000000000f03f010000000000f8bf"),
            ("cubic", 2, "float64", new long[] { 3 }, "000000000000f03ffffffffffffff3bf0fe08574f773b5bc"),
            ("cubic", 5, "float64", new long[] { 6 }, "000000000000f03f000000000000f4bf5555555555556dbc020000000000d03fabaaaaaaaa2ababcabaaaaaaaaead33c"),
            ("cubic", 12, "float64", new long[] { 13 }, "000000000000f03f000000000000f4bf142418e166f5bbbc070000000000d03f6f48f0d5273ab9bc52e6e8b3ee0fba3c2205f6c00cf4b7bc2774ad998c6dc23cf1b110ca1764c4bc145f8f6c1cb9bf3c8b7a686f436aa2bca92906709689c3bcb9f60e38c6eecc3c"),
            ("expi", 0, "complex128", new long[] { 1 }, "000000000000f03f0000000000000000"),
            ("expi", 1, "complex128", new long[] { 2 }, "21d12c7aec53e83f00000000000000000000000000000000f77f32bf3266ed3f"),
            ("expi", 2, "complex128", new long[] { 3 }, "846236aed77ce83f000000000000000000000000000000003ed05dacb825ec3ff6752647a10ccebf0000000000000000"),
            ("expi", 5, "complex128", new long[] { 6 }, "c095d7db7f7ce83f0000000000000000000000000000000016569aeec929ec3f7a5ed994506acdbf55555555555585bc55555555555565bcfd6967d06a08a4bf004f62d5b049743f555555555555753c55555555555585bc00df32b97177403f"),
            ("expi", 12, "complex128", new long[] { 13 }, "f0b8d7db7f7ce83f143bb1133bb1633c00000000000000006c0c97eec929ec3fef9bfa95506acdbf143bb1133bb1733c143bb1133bb1833ccdf738766a08a4bfbbf05a6be349743f0000000000000000143bb1133bb163bc9e787e483d5e403f8a9d99a197f405bf0000000000000000143bb1133bb163bcedc486b47034c9bec54e2c50fc4a893e143bb1133bb1633c143bb1133bb1733cc54eec889c8b463e3bb113fbd31302be143bb1133bb1633cd9899dd8899d88bc000000001858babdedc44eec64a0713d143bb1133bb1633c"),
            ("stack2", 0, "float64", new long[] { 1, 2 }, "0000000000000000000000000000f03f"),
            ("stack2", 1, "float64", new long[] { 2, 2 }, "000000000000000021d12c7aec53e83ff77f32bf3266ed3fb449d43dcc79733c"),
            ("stack2", 2, "float64", new long[] { 3, 2 }, "0000000000000000846236aed77ce83f3fd05dacb825ec3f60910db5b2687cbc00141e53690e40bcf7752647a10ccebf"),
            ("stack2", 5, "float64", new long[] { 6, 2 }, "0000000000000000c195d7db7f7ce83f17569aeec929ec3f2bf23622b7a55abc2b17338b5853673c795ed994506acdbff96967d06a08a4bf158038fc3ff470bcab14add374834bbcfe4e62d5b049743ffbdf32b97177403f700f038dbea068bc"),
            ("stack2", 12, "float64", new long[] { 13, 2 }, "0000000000000000f0b8d7db7f7ce83f6b0c97eec929ec3fc7935ff7bed7863c8a6545ad2dca72bceb9bfa95506acdbfd0f738766a08a4bfbc106a124d0a69bc76d2f74d8bbd123cdbf05a6be349743f7a797e483d5e403fc675b0a7447f643c823fbc373a5660bc5aa999a197f405bf909186b47034c9be36f9aeab684a48bca7f65047410058bce70a3750fc4a893e366ee3889c8b463ede5ba6d71bc748bc624fb2458e5728bc6356c8fcd31302be06a06a681558babd5483fa2b64fe58bc145b2cee837616bcda6909302ba0713d"),
            ("pyfloat", 0, "float64", new long[] { 1, 1 }, "0000000000000040"),
            ("pyfloat", 1, "float64", new long[] { 2, 2 }, "000000000000f03f000000000000f03fcd3b7f669ea0f6bfcd3b7f669ea0f63f"),
            ("pyfloat", 2, "float64", new long[] { 3, 3 }, "555555555555e53f555555555555e53f555555555555e53f1c339045a779f2bf00000000000000001c339045a779f23f535555555555e53f555555555555f5bf535555555555e53f"),
            ("i64", 0, "float64", new long[] { 1 }, "0000000000000000"),
            ("i64", 1, "float64", new long[] { 2 }, "00000000000000005354af998acc2340"),
            ("i64", 2, "float64", new long[] { 3 }, "00000000000000001c339045a77922400000000000000000"),
            ("i64", 5, "float64", new long[] { 6 }, "00000000000000005fd95b6b86e12240555555555555b53cabaaaaaaaaaadebc00000000000000001b5f6fca8560ddbf"),
            ("i64", 12, "float64", new long[] { 13 }, "00000000000000004317c2027c9d22400000000000000000b16210ab68a6c13f143bb1133bb1b3bc6aef5118b8cbd2bf9ed8899dd8899dbcc0db311952a1c6bf143bb1133bb1933cb44c3f98658ea0bf0000000000000000cf77989dcbb7963f0000000000000000"),
        };

        /// <summary>2**63, where a Python int becomes np.asarray's uint64.</summary>
        private static readonly BigInteger T = BigInteger.One << 63;

        /// <summary>
        ///     NumPy's outcome of <c>chebinterpolate(tanh_half, deg)</c> per deg kind (ci_pins.py's "deg kinds"): the result's
        ///     dtype and shape, or <c>!! PythonType: message</c>. The C# spelling follows the house map (a C# integer or
        ///     <see cref="BigInteger"/> is a Python int, <c>char</c> np.uint16, <see cref="Half"/> np.float16, an
        ///     <see cref="NDArray"/> an ndarray, <c>object[]</c> a list, a <c>ValueTuple</c> a tuple, null None).
        /// </summary>
        private static readonly (string Label, Func<object> Deg, string Outcome)[] DegKinds =
        {
            ("int 0", () => 0, "float64 (1,)"),
            ("int -1", () => -1, "!! ValueError: expected deg >= 0"),
            ("bool true", () => true, "!! TypeError: deg must be an int"),
            ("double 3.0", () => 3.0, "!! TypeError: deg must be an int"),
            ("str 3", () => "3", "!! TypeError: deg must be an int"),
            ("null", () => null, "!! TypeError: deg must be an int"),
            ("complex 3", () => new Complex(3, 0), "!! TypeError: deg must be an int"),
            ("half 3", () => (Half)3, "!! TypeError: deg must be an int"),
            ("char 3", () => (char)3, "float64 (4,)"),
            ("char 65535", () => (char)65535, "!! ValueError: npts must be >= 1"),
            ("nd0 int64 3", () => np.array(3L), "float64 (4,)"),
            ("nd0 uint8 255", () => np.array((byte)255), "!! ValueError: npts must be >= 1"),
            ("nd0 int8 127", () => np.array((sbyte)127), "!! ValueError: npts must be >= 1"),
            ("nd0 int8 -1", () => np.array((sbyte)-1), "!! ValueError: expected deg >= 0"),
            ("nd0 uint64 max", () => np.array(ulong.MaxValue), "!! ValueError: npts must be >= 1"),
            ("nd0 int16 3", () => np.array((short)3), "float64 (4,)"),
            ("nd0 bool true", () => np.array(true), "!! TypeError: deg must be an int"),
            ("nd0 float64 3", () => np.array(3.0), "!! TypeError: deg must be an int"),
            ("nd1 [3]", () => np.array(new long[] { 3 }), "!! TypeError: deg must be an int"),
            ("object[] {3}", () => new object[] { 3 }, "!! TypeError: deg must be an int"),
            ("object[] {}", () => new object[0], "!! TypeError: deg must be an int"),
            ("tuple (3,)", () => ValueTuple.Create(3), "!! TypeError: deg must be an int"),
            ("ragged", () => new object[] { new object[] { 1 }, new object[] { 2, 3 } },
                "!! ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part."),
            ("long max", () => long.MaxValue, "!! ValueError: npts must be >= 1"),
            ("big 2**63", () => T, "!! ValueError: Maximum allowed dimension exceeded"),
            ("big 2**64-1", () => ulong.MaxValue, "!! ValueError: npts must be >= 1"),
            ("big 2**64", () => BigInteger.One << 64, "!! TypeError: deg must be an int"),
            ("long min", () => long.MinValue, "!! ValueError: expected deg >= 0"),
            ("big -2**63-1", () => -T - 1, "!! TypeError: deg must be an int"),
            ("big 2**62", () => BigInteger.One << 62, "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("big 2**60", () => BigInteger.One << 60, "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
        };

        /// <summary>
        ///     NumPy's outcome of <c>chebinterpolate(lambda x, a: a * x, 3, args)</c> for args Python cannot unpack
        ///     (ci_pins.py's "args kinds").
        /// </summary>
        private static readonly (string Label, Func<object> Args, string Outcome)[] ArgsKinds =
        {
            ("int 5", () => 5, "!! TypeError: Value after * must be an iterable, not int"),
            ("double 2.0", () => 2.0, "!! TypeError: Value after * must be an iterable, not float"),
            ("bool true", () => true, "!! TypeError: Value after * must be an iterable, not bool"),
            ("nd0", () => np.array(2.0), "!! TypeError: iteration over a 0-d array"),
            ("char 3", () => (char)3, "!! TypeError: Value after * must be an iterable, not numpy.uint16"),
            ("half 2", () => (Half)2, "!! TypeError: Value after * must be an iterable, not numpy.float16"),
        };

        /// <summary>Leaves the process-global OpenBLAS engine off after every test (the LapackEigTests convention).</summary>
        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        // ---------------------------------------------------------------------------------------------
        //  NumPy's values
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     With the OpenBLAS backend — the products NumPy calls, at one thread — chebinterpolate is NumPy's dtype, shape and
        ///     float64 bytes for every pinned (function, degree): 1-D, 2-D and scalar results, real and complex, an integer
        ///     result promoted to float64.
        /// </summary>
        [TestMethod]
        public void Pins_WithTheBackend_AreNumPysBytes()
        {
            try
            {
                OpenBlasEngine.Enable(threads: 1);
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }
            foreach (var (fn, deg, dtype, shape, hex) in Pins)
            {
                NDArray r = np.polynomial.chebyshev.chebinterpolate(Functions[fn], deg);
                string label = $"{fn}({deg})";
                Assert.AreEqual(dtype, r.dtype.name, $"{label}: dtype");
                CollectionAssert.AreEqual(shape, r.shape, $"{label}: shape");
                Assert.AreEqual(hex, Hex(r), $"{label}: bytes");
            }
        }

        /// <summary>
        ///     Without a backend the managed product computes the same coefficients to rounding: every element within
        ///     <c>8 * 2**-52</c> times the largest |coefficient| of NumPy's (an absolute bound — the coefficients that are
        ///     exactly zero in exact arithmetic are pure rounding residue, so an ULP bound would be meaningless for them).
        /// </summary>
        [TestMethod]
        public void Pins_Managed_AreNumPysToRounding()
        {
            OpenBlasEngine.Disable();
            foreach (var (fn, deg, dtype, shape, hex) in Pins)
            {
                NDArray r = np.polynomial.chebyshev.chebinterpolate(Functions[fn], deg);
                string label = $"{fn}({deg})";
                Assert.AreEqual(dtype, r.dtype.name, $"{label}: dtype");
                CollectionAssert.AreEqual(shape, r.shape, $"{label}: shape");
                byte[] e = Convert.FromHexString(hex), a = r.tobytes();
                Assert.AreEqual(e.Length, a.Length, $"{label}: length");
                double scale = 0;
                for (int i = 0; i < e.Length; i += 8)
                    scale = System.Math.Max(scale, System.Math.Abs(BitConverter.ToDouble(e, i)));
                for (int i = 0; i < e.Length; i += 8)
                {
                    double d = System.Math.Abs(BitConverter.ToDouble(a, i) - BitConverter.ToDouble(e, i));
                    Assert.IsTrue(d <= 8 * System.Math.Pow(2, -52) * scale, $"{label}: element {i / 8} off by {d}");
                }
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  The arguments
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Every deg kind gives NumPy's outcome: <c>np.asarray(deg)</c> must be a 0-d integer array (a Python int, a char,
        ///     a 0-d integer NDArray), <c>deg + 1</c> wraps in deg's dtype (uint8 255, int8 127, char 65535, int64 2**63 - 1,
        ///     uint64 2**64 - 1: chebpts1's <c>npts must be &gt;= 1</c>), 2**63 is np.asarray's uint64 (an empty set of points,
        ///     then chebvander's dimension error), and a ragged list is np.asarray's ValueError.
        /// </summary>
        [TestMethod]
        public void DegKinds_MatchNumPy()
        {
            Func<NDArray, object[], object> f = (x, _) => np.tanh(x) + 0.5;
            foreach (var (label, deg, outcome) in DegKinds)
                Assert.AreEqual(outcome, Outcome(() => np.polynomial.chebyshev.chebinterpolate(f, deg())), $"deg {label}");
        }

        /// <summary>
        ///     args is unpacked as Python's <c>*args</c>: what Python cannot iterate is CPython's TypeError (the 0-d array its
        ///     iterator's own), raised before a null func's <c>'NoneType' object is not callable</c>; a deg error comes before
        ///     both.
        /// </summary>
        [TestMethod]
        public void ArgsKinds_AndANullFunc_MatchNumPy()
        {
            var cheb = np.polynomial.chebyshev;
            Func<NDArray, object[], object> scale = (x, a) => np.multiply(np.asanyarray(a[0]), x);
            foreach (var (label, args, outcome) in ArgsKinds)
                Assert.AreEqual(outcome, Outcome(() => cheb.chebinterpolate(scale, 3, args())), $"args {label}");
            Assert.AreEqual("!! TypeError: 'NoneType' object is not callable", Outcome(() => cheb.chebinterpolate(null, 3)));
            Assert.AreEqual("!! TypeError: Value after * must be an iterable, not int", Outcome(() => cheb.chebinterpolate(null, 3, 5)));
            Assert.AreEqual("!! ValueError: expected deg >= 0", Outcome(() => cheb.chebinterpolate(null, -1)));
        }

        /// <summary>
        ///     The unpacked args reach func as Python's <c>*args</c> would hand them over: a tuple's / list's items, an
        ///     NDArray's first-axis items (a 1-D array's elements, a 2-D array's rows as arrays), a string's characters as
        ///     one-character strings, a dictionary's keys, a multi-dimensional object array's rows as lists — and nothing for
        ///     null (NumPy's default <c>()</c>).
        /// </summary>
        [TestMethod]
        public void Args_ReachFuncAsPythonUnpacksThem()
        {
            object[] seen = null;
            Func<NDArray, object[], object> spy = (x, a) =>
            {
                seen = a;
                return x;
            };
            var cheb = np.polynomial.chebyshev;
            cheb.chebinterpolate(spy, 2);
            Assert.AreEqual(0, seen.Length, "null args");
            cheb.chebinterpolate(spy, 2, (1.5, "b"));
            CollectionAssert.AreEqual(new object[] { 1.5, "b" }, seen, "tuple");
            cheb.chebinterpolate(spy, 2, new List<object> { 7L, 8L });
            CollectionAssert.AreEqual(new object[] { 7L, 8L }, seen, "list");
            cheb.chebinterpolate(spy, 2, "ab");
            CollectionAssert.AreEqual(new object[] { "a", "b" }, seen, "str");
            cheb.chebinterpolate(spy, 2, new Dictionary<string, int> { ["k1"] = 1, ["k2"] = 2 });
            CollectionAssert.AreEqual(new object[] { "k1", "k2" }, seen, "dict keys");
            cheb.chebinterpolate(spy, 2, np.array(new[] { 2.0, 1.0 }));
            CollectionAssert.AreEqual(new object[] { 2.0, 1.0 }, seen, "1-D ndarray elements");
            cheb.chebinterpolate(spy, 2, np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }));
            Assert.AreEqual(2, seen.Length, "2-D ndarray rows");
            Assert.AreEqual("[3. 4.]", ((NDArray)seen[1]).ToString(), "2-D ndarray row 1");
            cheb.chebinterpolate(spy, 2, new object[,] { { 1, 2 }, { 3, 4 } });
            CollectionAssert.AreEqual(new object[] { 3, 4 }, (object[])seen[1], "object[,] rows as lists");
        }

        /// <summary>
        ///     func is called exactly once, with the deg + 1 Chebyshev points of the first kind (chebpts1's own float64 array),
        ///     and may return them: the identity interpolates T_1 (c = [0, 1, 0, …]).
        /// </summary>
        [TestMethod]
        public void Func_IsCalledOnceWithThePoints()
        {
            int calls = 0;
            string points = null;
            Func<NDArray, object[], object> identity = (x, _) =>
            {
                calls++;
                points = Hex(x);   // read inside the call: the points are an intermediate of it
                return x;
            };
            NDArray c = np.polynomial.chebyshev.chebinterpolate(identity, 6);
            Assert.AreEqual(1, calls);
            Assert.AreEqual(Hex(np.polynomial.chebyshev.chebpts1(7)), points, "the points are chebpts1(deg + 1)");
            Assert.AreEqual(1.0, c.GetDouble(1), 1e-15, "c[1]");
            for (int k = 0; k < 7; k++)
                if (k != 1)
                    Assert.AreEqual(0.0, c.GetDouble(k), 1e-15, $"c[{k}]");
        }

        /// <summary>
        ///     The ownership rule of the facade's remarks: the points handed to func and every array func creates are
        ///     intermediates of the call, reclaimed when it returns — unless func detaches one with
        ///     <see cref="NDScope.Detach(NDArray)"/> — while an array created before the call and returned by func stays the
        ///     caller's.
        /// </summary>
        [TestMethod]
        public void ArraysFuncCreates_AreReclaimedUnlessDetached()
        {
            NDArray handed = null, created = null, kept = null;
            NDArray constant = np.arange(4.0);
            Func<NDArray, object[], object> f = (x, _) =>
            {
                handed = x;
                created = x * 2.0;
                kept = x * 3.0;
                NDScope.Detach(kept);
                return constant;
            };
            NDArray c = np.polynomial.chebyshev.chebinterpolate(f, 3);
            Assert.IsTrue(handed.IsDisposed, "the points are reclaimed");
            Assert.IsTrue(created.IsDisposed, "an array func created is reclaimed");
            Assert.IsFalse(kept.IsDisposed, "a detached array survives");
            Assert.IsFalse(constant.IsDisposed, "the caller's array func returned survives");
            Assert.IsFalse(c.IsDisposed, "the result survives");
            Assert.AreEqual(4, kept.size);
            kept.Dispose();
        }

        /// <summary>
        ///     The result's forms are np.dot's: float64 for every real func result (float32, float16, integers, bool, a Python
        ///     int / float, a list, a tuple), complex128 for a complex one; <c>(deg + 1,)</c> for a vector result,
        ///     <c>(deg + 1, k)</c> for a <c>(deg + 1, k)</c> one, <c>(deg + 1, 2, 3)</c> for a <c>(2, deg + 1, 3)</c> one,
        ///     <c>(deg + 1, deg + 1)</c> for a scalar; a new C-contiguous array that owns its data.
        /// </summary>
        [TestMethod]
        public void ResultForms_AreNumPys()
        {
            var cheb = np.polynomial.chebyshev;
            var forms = new (string Label, Func<NDArray, object[], object> F, string Dtype, long[] Shape)[]
            {
                ("float32", (x, _) => np.sin(x).astype(np.float32), "float64", new long[] { 4 }),
                ("float16", (x, _) => np.sin(x).astype(np.float16), "float64", new long[] { 4 }),
                ("uint8", (x, _) => (np.abs(x) * 10).astype(np.uint8), "float64", new long[] { 4 }),
                ("bool", (x, _) => x > 0, "float64", new long[] { 4 }),
                ("python int", (x, _) => 2L, "float64", new long[] { 4, 4 }),
                ("list", (x, _) => Array.ConvertAll(np.sin(x).ToArray<double>(), v => (object)v), "float64", new long[] { 4 }),
                ("complex", (x, _) => new Complex(1, 2), "complex128", new long[] { 4, 4 }),
                ("2-D", (x, _) => np.stack(new[] { np.sin(x), np.cos(x) }, axis: 1), "float64", new long[] { 4, 2 }),
                ("3-D", (x, _) => np.multiply(np.sin(x).reshape(1, x.size, 1), np.ones(new Shape(2, 1, 3))), "float64", new long[] { 4, 2, 3 }),
            };
            foreach (var (label, f, dtype, shape) in forms)
            {
                NDArray c = cheb.chebinterpolate(f, 3);
                Assert.AreEqual(dtype, c.dtype.name, $"{label}: dtype");
                CollectionAssert.AreEqual(shape, c.shape, $"{label}: shape");
                Assert.IsTrue(c.Shape.IsContiguous && c.Storage.OwnsData, $"{label}: a new C-contiguous owner");
            }
        }

        /// <summary>
        ///     A func result np.dot refuses is NumPy's error: the wrong length (np.dot's alignment text), a ragged list
        ///     (np.array's inhomogeneous-shape text); and one NumPy computes with as Python objects (null) is refused.
        /// </summary>
        [TestMethod]
        public void RefusedResults_MatchNumPy()
        {
            var cheb = np.polynomial.chebyshev;
            Assert.AreEqual("!! ValueError: shapes (4,4) and (5,) not aligned: 4 (dim 1) != 5 (dim 0)",
                Outcome(() => cheb.chebinterpolate((x, _) => np.ones(new Shape(x.size + 1)), 3)));
            Assert.AreEqual("!! ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (4,) + inhomogeneous part.",
                Outcome(() => cheb.chebinterpolate((x, _) => new object[]
                {
                    new object[] { 1.0 }, new object[] { 1.0, 2.0 }, new object[] { 1.0, 2.0 }, new object[] { 1.0, 2.0 },
                }, 3)));
            Assert.ThrowsException<NotSupportedException>(() => cheb.chebinterpolate((x, _) => null, 3));
        }

        /// <summary>
        ///     A <c>numpy.matrix</c> func result is computed as a plain array (<c>[Misaligned]</c>: NumPy's np.dot returns the
        ///     matrix subclass, so its chebinterpolate returns a matrix): the numpy.polynomial package computes plain, and the
        ///     values are the plain array's.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void MatrixResult_IsComputedPlain()
        {
            var cheb = np.polynomial.chebyshev;
            Func<NDArray, object[], object> plain = (x, _) => np.stack(new[] { np.sin(x), np.cos(x) }, axis: 1);
            Func<NDArray, object[], object> matrix = (x, _) => np.asmatrix(np.stack(new[] { np.sin(x), np.cos(x) }, axis: 1));
            NDArray want = cheb.chebinterpolate(plain, 5), got = cheb.chebinterpolate(matrix, 5);
            Assert.IsNotInstanceOfType(got, typeof(NDMatrix), "a plain array");
            CollectionAssert.AreEqual(new long[] { 6, 2 }, got.shape);
            Assert.AreEqual(Hex(want), Hex(got), "the plain array's values");
        }

        // ---------------------------------------------------------------------------------------------
        //  The composition
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     chebinterpolate IS NumPy's statements: for every degree 0..60, a real and a complex function, the bytes equal
        ///     <c>c = np.dot(chebvander(x, deg).T, f(x)); c[0] /= deg + 1; c[1:] /= 0.5*(deg + 1)</c> composed by hand from the
        ///     same public functions (whatever product is installed — both sides call the same np.dot). The complex divisions
        ///     are NumPy's CDOUBLE_divide (a multiplication by the reciprocal), the hand composition's as well.
        /// </summary>
        [TestMethod]
        public void Result_IsNumPysStatementsComposed()
        {
            var cheb = np.polynomial.chebyshev;
            var fs = new Func<NDArray, object[], object>[]
            {
                (x, _) => np.exp(x) * 3.7,
                (x, _) => np.exp(np.multiply(NDArray.Scalar(new Complex(0, 1.3)), x)),
            };
            foreach (var f in fs)
                for (int deg = 0; deg <= 60; deg++)
                {
                    NDArray x = cheb.chebpts1(deg + 1);
                    NDArray m = cheb.chebvander(x, deg);
                    NDArray c = np.dot(m.T, (NDArray)f(x, Array.Empty<object>()));
                    NDArray head = c[0], tail = c[new Slice(1, null)];
                    np.divide(head, NDArray.Scalar(deg + 1L), @out: head);
                    np.divide(tail, NDArray.Scalar(0.5 * (deg + 1)), @out: tail);
                    Assert.AreEqual(Hex(c), Hex(cheb.chebinterpolate(f, deg)), $"degree {deg}");
                }
        }

        // ---------------------------------------------------------------------------------------------
        //  NumPy's own tests (numpy/polynomial/tests/test_chebyshev.py, TestInterpolate)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>TestInterpolate</c>: <c>test_raises</c> (ValueError for -1, TypeError for 10.), <c>test_dimensions</c>
        ///     (shape (deg + 1,) for degrees 1..4), and <c>test_approximation</c> (the interpolant of <c>x**p</c> through
        ///     <c>args=(p,)</c> reproduces it to 12 decimals at 10 points, for every p ≤ deg &lt; 10).
        /// </summary>
        [TestMethod]
        public void NumPysOwnTests()
        {
            var cheb = np.polynomial.chebyshev;
            Func<NDArray, object[], object> f = (x, _) => x * (x - 1) * (x - 2);
            Assert.ThrowsException<ValueError>(() => cheb.chebinterpolate(f, -1));
            Assert.ThrowsException<TypeError>(() => cheb.chebinterpolate(f, 10.0));
            for (int deg = 1; deg < 5; deg++)
                CollectionAssert.AreEqual(new long[] { deg + 1 }, cheb.chebinterpolate(f, deg).shape, $"degree {deg}");

            Func<NDArray, object[], object> powx = (x, a) => np.power(x, Convert.ToInt64(a[0]));
            NDArray pts = np.linspace(-1.0, 1.0, 10);
            for (int deg = 0; deg < 10; deg++)
                for (int p = 0; p <= deg; p++)
                {
                    NDArray c = cheb.chebinterpolate(powx, deg, new object[] { (long)p });
                    double err = np.max(np.abs(cheb.chebval(pts, c) - np.power(pts, (long)p))).GetDouble();
                    Assert.IsTrue(err < 1.5e-12, $"x**{p} at degree {deg}: off by {err}");
                }
        }

        // ---------------------------------------------------------------------------------------------
        //  Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     A call's outcome in the probe's text: the result's dtype and shape (Python's tuple text), or
        ///     <c>!! PythonType: message</c> — NumSharp's exceptions named after NumPy's, an ArgumentException /
        ///     IncorrectShapeException as ValueError (np.dot's alignment errors).
        /// </summary>
        /// <param name="call">The call.</param>
        /// <returns>The text.</returns>
        private static string Outcome(Func<NDArray> call)
        {
            try
            {
                NDArray r = call();
                return $"{r.dtype.name} {r.Shape.ToPythonTuple()}";
            }
            catch (ValueError e) { return "!! ValueError: " + e.Message; }
            catch (TypeError e) { return "!! TypeError: " + e.Message; }
            catch (IncorrectShapeException e) { return "!! ValueError: " + e.Message; }
            catch (ArgumentException e) { return "!! ValueError: " + e.Message; }
        }

        /// <summary>The C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a) => Convert.ToHexString(a.tobytes()).ToLowerInvariant();
    }
}
