using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> companion matrices and roots (plan U7): <c>{p}companion</c> and <c>{p}roots</c> of the six
    ///     bases. Three kinds of test live here: NumPy 2.4.2's own <c>TestCompanion</c> / <c>test_{p}roots</c> ported; byte dumps
    ///     NumPy produced (the probe inputs are spelled out next to each test — the float32 / float16 dumps pin the dtype rule:
    ///     the cheb / leg / herm / herme last column computed in float64 and rounded once, the power / Laguerre forms staying
    ///     in the series' dtype); and what the oracle (<c>polyroots.jsonl</c> / <c>polyroots_parity.jsonl</c>) cannot see — the
    ///     object-series refusals NumPy answers with object arrays, decimal and char, the backend-missing path, the
    ///     write-once allocation of a large companion and the int64 ramp kernel.
    /// </summary>
    /// <remarks>
    ///     Roots of degree 2 or more need a LAPACK-capable library (the bundled scipy-openblas is one); those tests go
    ///     <see cref="Assert.Inconclusive(string)"/> on a host without one, as <see cref="Backends.LapackEigTests"/> does.
    ///     Their VALUES are asserted to a tolerance (the bits depend on the OpenBLAS build, thread count and kernel — byte
    ///     parity is the host-pinned <c>PolyrootsParity</c> oracle tier); everything before LAPACK is asserted bit for bit.
    /// </remarks>
    [TestClass]
    public class PolynomialRootsTests
    {
        /// <summary>One basis's companion / roots pair plus what NumPy's tests build them from.</summary>
        /// <param name="Name">The basis prefix NumPy uses (<c>poly</c>, <c>cheb</c>, ...).</param>
        /// <param name="Companion"><c>{p}companion(c)</c>.</param>
        /// <param name="Roots"><c>{p}roots(c)</c>.</param>
        /// <param name="FromRoots"><c>{p}fromroots(roots)</c>.</param>
        /// <param name="LinearCompanion">NumPy's <c>test_linear_root</c>: <c>{p}companion([1, 2])[0, 0]</c>.</param>
        /// <param name="LinearSeries">The two-term series NumPy's <c>test_{p}roots</c> passes.</param>
        /// <param name="LinearRoot">Its one root.</param>
        /// <param name="RootLow">The low end of NumPy's round-trip <c>np.linspace(low, high, i)</c>.</param>
        /// <param name="RootHigh">The high end.</param>
        private sealed record Basis(string Name, Func<object, NDArray> Companion, Func<object, NDArray> Roots,
            Func<object, NDArray> FromRoots, double LinearCompanion, int[] LinearSeries, double LinearRoot, double RootLow,
            double RootHigh);

        /// <summary>The six bases, in NumPy's module order, with NumPy's own test constants.</summary>
        private static readonly Basis[] Bases =
        {
            new("poly", np.polynomial.polynomial.polycompanion, np.polynomial.polynomial.polyroots,
                np.polynomial.polynomial.polyfromroots, -0.5, new[] { 1, 2 }, -0.5, -1, 1),
            new("cheb", np.polynomial.chebyshev.chebcompanion, np.polynomial.chebyshev.chebroots,
                np.polynomial.chebyshev.chebfromroots, -0.5, new[] { 1, 2 }, -0.5, -1, 1),
            new("leg", np.polynomial.legendre.legcompanion, np.polynomial.legendre.legroots,
                np.polynomial.legendre.legfromroots, -0.5, new[] { 1, 2 }, -0.5, -1, 1),
            new("lag", np.polynomial.laguerre.lagcompanion, np.polynomial.laguerre.lagroots,
                np.polynomial.laguerre.lagfromroots, 1.5, new[] { 0, 1 }, 1, 0, 3),
            new("herm", np.polynomial.hermite.hermcompanion, np.polynomial.hermite.hermroots,
                np.polynomial.hermite.hermfromroots, -0.25, new[] { 1, 1 }, -0.5, -1, 1),
            new("herme", np.polynomial.hermite_e.hermecompanion, np.polynomial.hermite_e.hermeroots,
                np.polynomial.hermite_e.hermefromroots, -0.5, new[] { 1, 1 }, -1, -1, 1),
        };

        /// <summary>Leaves the process-global OpenBLAS engine off after every test (the LapackEigTests convention).</summary>
        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        /// <summary>Installs the LAPACK backend, or goes Inconclusive on a host that cannot load one.</summary>
        private static void RequireLapack()
        {
            try
            {
                OpenBlasEngine.Enable(threads: 1);
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }

            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("the loaded BLAS exports no LAPACK routines (a bare reference CBLAS).");
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

        /// <summary>A NumPy error's type and verbatim text.</summary>
        /// <typeparam name="TException">The expected exception type.</typeparam>
        /// <param name="call">The call.</param><param name="text">NumPy's message.</param>
        /// <param name="because">Context for the failure message.</param>
        private static void Raises<TException>(Action call, string text, string because = "") where TException : Exception
            => call.Should().Throw<TException>(because).Which.Message.Should().StartWith(text, because);

        /// <summary>NumPy's <c>assert_almost_equal(actual, desired, decimal)</c> over float64 values.</summary>
        /// <param name="actual">The values.</param><param name="desired">The expected values.</param>
        /// <param name="decimals">NumPy's <c>decimal</c> (tolerance <c>1.5 * 10**-decimal</c>).</param>
        /// <param name="because">Context.</param>
        private static void AlmostEqual(NDArray actual, double[] desired, int decimals, string because)
        {
            actual.size.Should().Be(desired.Length, because);
            var a = actual.astype(np.float64).ToArray<double>();
            double tol = 1.5 * System.Math.Pow(10, -decimals);
            for (int i = 0; i < a.Length; i++)
                System.Math.Abs(a[i] - desired[i]).Should().BeLessThan(tol, $"{because} [{i}]");
        }

        // =========================================================================================================
        //  NumPy 2.4.2's own TestCompanion / test_{p}roots (numpy/polynomial/tests/test_*.py), ported
        // =========================================================================================================

        /// <summary>
        ///     <c>TestCompanion</c>: <c>{p}companion([])</c> and <c>{p}companion([1])</c> raise ValueError, the companion of
        ///     <c>[0]*i + [1]</c> is (i, i), and the linear series <c>[1, 2]</c> gives the 1x1 matrix of its root (poly / cheb /
        ///     leg / herme -0.5, lag 1.5, herm -0.25 — <c>-.5 * c[0] / c[1]</c>).
        /// </summary>
        [TestMethod]
        public void Companion_RaisesDimensionsLinearRoot_NumPyTestCompanion()
        {
            foreach (var b in Bases)
            {
                Raises<ValueError>(() => b.Companion(new object[0]), "Coefficient array is empty", b.Name);
                Raises<ValueError>(() => b.Companion(new object[] { 1 }), "Series must have maximum degree of at least 1.", b.Name);
                for (int i = 1; i < 5; i++)
                {
                    var coef = Enumerable.Repeat((object)0, i).Append(1).ToArray();
                    b.Companion(coef).shape.Should().Equal(new long[] { i, i }, $"{b.Name} degree {i}");
                }
                var lin = b.Companion(new object[] { 1, 2 });
                lin.shape.Should().Equal(new long[] { 1, 1 }, b.Name);
                lin.GetDouble(0, 0).Should().Be(b.LinearCompanion, b.Name);
            }
        }

        /// <summary>
        ///     <c>test_{p}roots</c>: a constant series has no roots, the linear series has its one root, and <c>{p}roots</c> of
        ///     <c>{p}fromroots(np.linspace(low, high, i))</c> recovers the points (decimal = 7) for i = 2..4.
        /// </summary>
        [TestMethod]
        public void Roots_ConstantLinearAndRoundTrip_NumPyTestMisc()
        {
            RequireLapack();
            foreach (var b in Bases)
            {
                b.Roots(new object[] { 1 }).size.Should().Be(0, b.Name);
                AlmostEqual(b.Roots(b.LinearSeries.Cast<object>().ToArray()), new[] { b.LinearRoot }, 7, b.Name);
                for (int i = 2; i < 5; i++)
                {
                    var tgt = np.linspace(b.RootLow, b.RootHigh, i);
                    AlmostEqual(b.Roots(b.FromRoots(tgt)), tgt.ToArray<double>(), 7, $"{b.Name} round trip {i}");
                }
            }
        }

        /// <summary>
        ///     <c>test_polyroots</c>' "larger root values": for 1,000 roots <c>i</c> in <c>np.logspace(10, 25, 1000)</c>,
        ///     <c>polyroots(polyfromroots([-1, 1, i]))</c> recovers <c>[-1, 1, i]</c> to <c>15 - int(log10(i))</c> decimals.
        /// </summary>
        [TestMethod]
        public void PolyRoots_LargerRootValues_NumPyTestPolyroots()
        {
            RequireLapack();
            foreach (double i in np.logspace(10, 25, 1000).ToArray<double>())
            {
                var tgt = new[] { -1.0, 1.0, i };
                var res = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyfromroots(tgt));
                AlmostEqual(res, tgt, 15 - (int)System.Math.Log10(i), $"root {i}");
            }
        }

        // =========================================================================================================
        //  NumPy byte dumps
        // =========================================================================================================

        /// <summary>
        ///     <c>{p}companion</c> of <c>[1, -2, 0.5, 3]</c> as float32 and float16, of <c>[1+1j, -2, 0.5j, 3]</c> as complex128
        ///     and of the int64 series <c>[4, -1, 2, 7]</c> (computed in float64) — NumPy 2.4.2's bytes. The narrow dtypes pin
        ///     the dtype rule: cheb / leg / herm / herme compute the last column in float64 (their helper vectors are float64)
        ///     and round once into the matrix, while poly / lag stay in the series' dtype.
        /// </summary>
        [TestMethod]
        public void Companion_EveryBasisAndDtype_IsNumPysBytes()
        {
            var dumps = new (string name, string f32, string f16, string c128, string i64)[]
            {
                ("poly",
                    "0000000000000000abaaaabe0000803f00000000abaa2a3f000000000000803fabaa2abe",
                    "0000000055b5003c000055390000003c55b1",
                    "0000000000000000000000000000000000000000000000000000000000000000555555555555d5bf555555555555d5bf000000000000f03f000000000000000000000000000000000000000000000000555555555555e53f000000000000000000000000000000000000000000000000000000000000f03f00000000000000000000000000000000555555555555c5bf",
                    "00000000000000000000000000000000922449922449e2bf000000000000f03f0000000000000000922449922449c23f0000000000000000000000000000f03f922449922449d2bf"),
                ("cheb",
                    "00000000f304353fef5b71bef304353f000000005655553f000000000000003fabaaaabd",
                    "0000a8398ab3a8390000aa3a0000003855ad",
                    "00000000000000000000000000000000cd3b7f669ea0e63f000000000000000065fafedd7d2bcebf65fafedd7d2bcebfcd3b7f669ea0e63f000000000000000000000000000000000000000000000000aaaaaaaaaaaaea3f000000000000000000000000000000000000000000000000000000000000e03f00000000000000000000000000000000555555555555b5bf",
                    "0000000000000000cd3b7f669ea0e63f564448be22dcd9bfcd3b7f669ea0e63f0000000000000000922449922449e23f0000000000000000000000000000e03f922449922449c2bf"),
                ("leg",
                    "000000003acd133f2ff9e4be3acd133f00000000a532843f00000000a532043fcdccccbd",
                    "00009e3827b79e380000223c0000223866ae",
                    "000000000000000000000000000000001d339045a779e23f0000000000000000d9edbfc5259fdcbfd9edbfc5259fdcbf1d339045a779e23f000000000000000000000000000000000000000000000000dbf6d4a25486f03f000000000000000000000000000000000000000000000000dbf6d4a25486e03f00000000000000000000000000000000999999999999b9bf",
                    "00000000000000001d339045a779e23f4da7ed846988e8bf1d339045a779e23f00000000000000009ce2947cd410e43f0000000000000000dbf6d4a25486e03f155ff1155ff1c5bf"),
                ("lag",
                    "0000803f000080bf0000803f000080bf00004040000080c000000000000000c00000b040",
                    "003c00bc003c00bc004200c4000000c08045",
                    "000000000000f03f0000000000000000000000000000f0bf0000000000000000000000000000f03f000000000000f03f000000000000f0bf00000000000000000000000000000840000000000000000000000000000010c000000000000000000000000000000000000000000000000000000000000000c000000000000000000000000000001440000000000000e03f",
                    "000000000000f03f000000000000f0bfdbb66ddbb66dfb3f000000000000f0bf0000000000000840dbb66ddbb66d03c0000000000000000000000000000000c0dbb66ddbb66d1740"),
                ("herm",
                    "00000000f304353fef5b71bdf304353f000000005555953f000000000000803fabaaaabd",
                    "0000a8398baba8390000ab3c0000003c55ad",
                    "00000000000000000000000000000000cd3b7f669ea0e63f000000000000000065fafedd7d2baebf65fafedd7d2baebfcd3b7f669ea0e63f000000000000000000000000000000000000000000000000abaaaaaaaaaaf23f000000000000000000000000000000000000000000000000000000000000f03f00000000000000000000000000000000555555555555b5bf",
                    "0000000000000000cd3b7f669ea0e63f574448be22dcb9bfcd3b7f669ea0e63f0000000000000000254992244992f03f0000000000000000000000000000f03f922449922449c2bf"),
                ("herme",
                    "000000000000803fef5b71be0000803f00000000ef5bf13f00000000f304b53fabaa2abe",
                    "0000003c8bb3003c00008b3f0000a83d55b1",
                    "00000000000000000000000000000000000000000000f03f000000000000000065fafedd7d2bcebf65fafedd7d2bcebf000000000000f03f00000000000000000000000000000000000000000000000066fafedd7d2bfe3f000000000000000000000000000000000000000000000000cd3b7f669ea0f63f00000000000000000000000000000000555555555555c5bf",
                    "0000000000000000000000000000f03f574448be22dcd9bf000000000000f03f000000000000000012c06392603ef83f0000000000000000cd3b7f669ea0f63f922449922449d2bf"),
            };
            var shape = new long[] { 3, 3 };
            foreach (var (b, d) in Bases.Zip(dumps))
            {
                d.name.Should().Be(b.Name);
                AssertBytes(b.Companion(np.array(new float[] { 1, -2, 0.5f, 3 })), "float32", shape, d.f32, $"{b.Name} float32");
                AssertBytes(b.Companion(np.array(new Half[] { (Half)1, (Half)(-2), (Half)0.5, (Half)3 })), "float16", shape, d.f16,
                    $"{b.Name} float16");
                AssertBytes(b.Companion(np.array(new[] { new Complex(1, 1), new Complex(-2, 0), new Complex(0, 0.5), new Complex(3, 0) })),
                    "complex128", shape, d.c128, $"{b.Name} complex128");
                AssertBytes(b.Companion(np.array(new long[] { 4, -1, 2, 7 })), "float64", shape, d.i64, $"{b.Name} int64");
            }
        }

        /// <summary>
        ///     A two-term series' root is NumPy's scalarmath on the series' scalars, returned as a 1-element array of their
        ///     dtype: <c>-c[0] / c[1]</c> (lag <c>1 + c[0] / c[1]</c>, herm <c>-.5 * c[0] / c[1]</c>) — no LAPACK. NumPy 2.4.2:
        ///     <c>[1, 2]</c> int64, <c>[1.5, 4]</c> float32, <c>[2, 1j]</c> complex128 (whose negated real part is -0.0); a
        ///     trailing -0.0 is trimmed, leaving a constant: an empty float64 array.
        /// </summary>
        [TestMethod]
        public void Roots_LinearSeries_AreNumPysScalarmath()
        {
            var dumps = new (string name, string i64, string f32, string c128)[]
            {
                ("poly", "000000000000e0bf", "0000c0be", "00000000000000800000000000000040"),
                ("cheb", "000000000000e0bf", "0000c0be", "00000000000000800000000000000040"),
                ("leg", "000000000000e0bf", "0000c0be", "00000000000000800000000000000040"),
                ("lag", "000000000000f83f", "0000b03f", "000000000000f03f00000000000000c0"),
                ("herm", "000000000000d0bf", "000040be", "0000000000000000000000000000f03f"),
                ("herme", "000000000000e0bf", "0000c0be", "00000000000000800000000000000040"),
            };
            var one = new long[] { 1 };
            foreach (var (b, d) in Bases.Zip(dumps))
            {
                d.name.Should().Be(b.Name);
                AssertBytes(b.Roots(np.array(new long[] { 1, 2 })), "float64", one, d.i64, $"{b.Name} int64");
                AssertBytes(b.Roots(np.array(new float[] { 1.5f, 4f })), "float32", one, d.f32, $"{b.Name} float32");
                AssertBytes(b.Roots(np.array(new[] { new Complex(2, 0), new Complex(0, 1) })), "complex128", one, d.c128,
                    $"{b.Name} complex128");
                AssertBytes(b.Roots(np.array(new double[] { 3.0, -0.0 })), "float64", new long[] { 0 }, "", $"{b.Name} trimmed");
            }
        }

        /// <summary>
        ///     A float32 series whose roots are complex: NumPy returns complex64 (eigvals' <c>astype(complex64)</c>), NumSharp
        ///     complex128 holding exactly those values (#569) — sorted lexicographically like NumPy's. NumPy 2.4.2:
        ///     <c>polyroots(np.array([1, 0.5, 0.25, 1], np.float32)).astype(np.complex128)</c>.
        /// </summary>
        [TestMethod]
        public void Roots_Float32ComplexRoots_AreNumPysComplex64Values()
        {
            RequireLapack();
            var r = np.polynomial.polynomial.polyroots(np.array(new float[] { 1, 0.5f, 0.25f, 1 }));
            AssertBytes(r, "complex128", new long[] { 3 },
                "000000c0b519edbf0000000000000000000000c0b519d53f00000040d0daefbf000000c0b519d53f00000040d0daef3f");
        }

        // =========================================================================================================
        //  Errors, in NumPy's order
        // =========================================================================================================

        /// <summary>
        ///     NumPy 2.4.2's errors, both functions, every basis: as_series' empty / not-1-d / no-common-type ValueErrors (a str
        ///     or bool series), companion's degree check, and — for roots of degree 2 or more — eigvals' checks in ITS order:
        ///     finiteness (LinAlgError) before the dtype (TypeError), so a float16 series holding a NaN reports the NaN. None of
        ///     these needs a backend: each is raised before LAPACK.
        /// </summary>
        [TestMethod]
        public void Errors_AreNumPysInNumPysOrder()
        {
            OpenBlasEngine.Disable();
            foreach (var b in Bases)
            {
                foreach (var f in new[] { b.Companion, b.Roots })
                {
                    Raises<ValueError>(() => f(new object[0]), "Coefficient array is empty", b.Name);
                    Raises<ValueError>(() => f(new object[] { new object[] { 1, 2 } }), "Coefficient array is not 1-d", b.Name);
                    Raises<ValueError>(() => f(new object[] { "a", 1 }), "Coefficient arrays have no common type", b.Name);
                    Raises<ValueError>(() => f(new[] { true, false }), "Coefficient arrays have no common type", b.Name);
                }
                Raises<ValueError>(() => b.Companion(new object[] { 1 }), "Series must have maximum degree of at least 1.", b.Name);
                b.Roots(new object[] { 1 }).dtype.name.Should().Be("float64", b.Name);

                Raises<TypeError>(() => b.Roots(np.array(new Half[] { (Half)1, (Half)2, (Half)3 })),
                    "array type float16 is unsupported in linalg", b.Name);
                Raises<LinAlgError>(() => b.Roots(np.array(new[] { 1.0, double.NaN, 3.0 })),
                    "Array must not contain infs or NaNs", b.Name);
                Raises<LinAlgError>(() => b.Roots(np.array(new Half[] { (Half)1, Half.NaN, (Half)3 })),
                    "Array must not contain infs or NaNs", b.Name);
            }
        }

        /// <summary>
        ///     NumPy keeps going in the OBJECT dtype for a series holding None (or a Python int past uint64), and companion's
        ///     next statement is the length check on the TRIMMED object array: <c>[None, 0]</c> and <c>[None]</c> raise its
        ///     ValueError, which NumSharp reproduces. Past it NumPy computes with Python objects, a dtype NumSharp lacks.
        /// </summary>
        [TestMethod]
        public void Companion_ObjectSeries_LengthCheckOnTheTrimmedSeries()
        {
            foreach (var b in Bases)
            {
                foreach (var c in new[] { new object[] { null, 0 }, new object[] { null }, new object[] { null, 0.0, -0.0 },
                             new object[] { BigInteger.Pow(2, 70), 0 } })
                    Raises<ValueError>(() => b.Companion(c), "Series must have maximum degree of at least 1.", b.Name);
                b.Invoking(x => x.Companion(new object[] { null, 1 })).Should().Throw<NotSupportedException>(b.Name);
            }
        }

        /// <summary>
        ///     DIVERGENCE: NumPy's <c>{p}roots([None])</c> and <c>{p}roots([None, 0])</c> return <c>np.array([], dtype=object)</c>
        ///     — an object array, a dtype NumSharp does not have — so NumSharp refuses every object series' roots with
        ///     NotSupportedException rather than return a numeric empty array NumPy does not.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Roots_ObjectSeries_AreRefused()
        {
            foreach (var b in Bases)
            {
                b.Invoking(x => x.Roots(new object[] { null })).Should().Throw<NotSupportedException>(b.Name);
                b.Invoking(x => x.Roots(new object[] { null, 0 })).Should().Throw<NotSupportedException>(b.Name);
            }
        }

        /// <summary>
        ///     Without a backend, a series of degree 2 or more reaches np.linalg.eigvals, which NumSharp.Core cannot compute:
        ///     MissingBackendException naming the eigensolver. The constant and linear short cuts and the companion matrix need
        ///     none.
        /// </summary>
        [TestMethod]
        public void Roots_WithoutBackend_DegreeTwoOrMoreRaisesMissingBackend()
        {
            OpenBlasEngine.Disable();
            foreach (var b in Bases)
            {
                b.Invoking(x => x.Roots(new object[] { 1, 2, 3 })).Should().Throw<MissingBackendException>(b.Name);
                b.Roots(new object[] { 1, 2 }).size.Should().Be(1, b.Name);
                b.Companion(new object[] { 1, 2, 3 }).shape.Should().Equal(new long[] { 2, 2 }, b.Name);
            }
        }

        // =========================================================================================================
        //  Result layout, decimal and char
        // =========================================================================================================

        /// <summary>
        ///     NumPy's result objects: the companion is a fresh C-contiguous owning matrix; real roots of a float64 series are a
        ///     VIEW of the real parts of eigvals' complex result (<c>w.real</c>, kept by <c>astype(copy=False)</c>, sorted in
        ///     place) — not contiguous, not owning, stride 16 bytes; a float32 series' real roots are a fresh cast.
        /// </summary>
        [TestMethod]
        public void Roots_RealRootsAreAViewOfTheComplexEigenvalues()
        {
            RequireLapack();
            foreach (var b in Bases)
            {
                var c = b.FromRoots(new[] { 1.0, -2.0, 0.5 });
                var k = b.Companion(c);
                (k.flags.c_contiguous && k.flags.owndata).Should().BeTrue(b.Name);

                var r = b.Roots(c);
                r.dtype.name.Should().Be("float64", b.Name);
                r.flags.c_contiguous.Should().BeFalse(b.Name);
                r.flags.f_contiguous.Should().BeFalse(b.Name);
                r.flags.owndata.Should().BeFalse(b.Name);
                r.strides.Should().Equal(new long[] { 16 }, b.Name);
                AlmostEqual(r, new[] { -2.0, 0.5, 1.0 }, 7, b.Name);

                var r32 = b.Roots(c.astype(np.float32));
                r32.dtype.name.Should().Be("float32", b.Name);
                (r32.flags.c_contiguous && r32.flags.owndata).Should().BeTrue(b.Name);
            }
        }

        /// <summary>
        ///     Decimal (no NumPy analog) runs the same statements in decimal arithmetic — the float64 helper vectors promote
        ///     into decimal, the dtype NumSharp's result_type gives (decimal, float64) — and its roots of degree 2 or more take
        ///     linalg's float16 exit; a char series is computed exactly as uint16 is.
        /// </summary>
        [TestMethod]
        public void Companion_DecimalAndChar()
        {
            var poly = np.polynomial.polynomial.polycompanion(np.array(new decimal[] { 1m, -2m, 0.5m, 3m }));
            poly.dtype.name.Should().Be("decimal");
            poly.ToArray<decimal>().Should().Equal(0m, 0m, 1m / -3m, 1m, 0m, -2m / -3m, 0m, 1m, 0.5m / -3m);
            np.polynomial.laguerre.lagcompanion(np.array(new decimal[] { 1m, -2m, 0.5m, 3m })).GetDecimal(2, 2)
                .Should().Be(5m + 0.5m / 3m * 3m);

            foreach (var b in Bases)
            {
                b.Companion(np.array(new decimal[] { 1m, -2m, 0.5m, 3m })).dtype.name.Should().Be("decimal", b.Name);
                Raises<TypeError>(() => b.Roots(np.array(new decimal[] { 1m, -2m, 3m })), "array type decimal is unsupported in linalg",
                    b.Name);
                b.Roots(np.array(new decimal[] { 3m, 4m })).dtype.name.Should().Be("decimal", b.Name);

                var ch = b.Companion(np.array(new[] { (char)4, (char)1, (char)2, (char)7 }));
                np.array_equal(ch, b.Companion(np.array(new ushort[] { 4, 1, 2, 7 }))).Should().BeTrue(b.Name);
                b.Roots(np.array(new[] { (char)5 })).dtype.name.Should().Be("float64", b.Name);
            }
        }

        // =========================================================================================================
        //  Allocation and kernels
        // =========================================================================================================

        /// <summary>
        ///     The companion's zero matrix is a write-once allocation (np.tri's policy): up to 64 MiB a pooled buffer memset to
        ///     zero — so a reused DIRTY buffer must come back clean — and above it the OS's zeroed pages. Both paths hold only
        ///     the sub-diagonal and the last column: <c>2n - 1</c> nonzeros for a series whose quotients are all nonzero.
        /// </summary>
        [TestMethod]
        public void Companion_LargeMatrices_BothAllocationPathsAreZeroed()
        {
            // Pooled path: leave a NaN-filled buffer of the same size in the pool first.
            const int n = 128;
            var dirty = new NDArray(NPTypeCode.Double, new Shape(n, n), false);
            dirty.fill(double.NaN);
            dirty.Dispose();
            var c = np.arange(n + 1).astype(np.float64) + 1.0;
            var k = np.polynomial.polynomial.polycompanion(c);
            np.count_nonzero(k).Should().Be(2 * n - 1);
            k.GetDouble(1, 0).Should().Be(1.0);
            k.GetDouble(0, n - 1).Should().Be(-1.0 / (n + 1));

            // OS-zeroed path: 2897^2 float64 = 67,141,672 bytes, past np.WriteOnceMaxBytes.
            const int big = 2897;
            ((long)big * big * 8).Should().BeGreaterThan(np.WriteOnceMaxBytes);
            using var cb = np.ones(new Shape(big + 1), NPTypeCode.Double);
            using var kb = np.polynomial.polynomial.polycompanion(cb);
            np.count_nonzero(kb).Should().Be(2 * big - 1);
            kb.GetDouble(big - 1, big - 2).Should().Be(1.0);
            kb.GetDouble(big - 1, big - 1).Should().Be(-1.0);
        }

        /// <summary>
        ///     The int64 ramp kernel behind np.arange's role in the companion statements: <c>dst[i] = start + i * step</c> for
        ///     i &lt; n, written as a running sum; a zero or negative count writes nothing.
        /// </summary>
        [TestMethod]
        public unsafe void RampKernel_WritesTheRampAndNothingForNonPositiveCounts()
        {
            var ramp = DirectILKernelGenerator.GetPolyRampKernel();
            var buf = stackalloc long[8];
            for (int i = 0; i < 8; i++) buf[i] = 777;
            ramp(buf, 5, 9, -2);
            new[] { buf[0], buf[1], buf[2], buf[3], buf[4], buf[5] }.Should().Equal(9, 7, 5, 3, 1, 777);
            for (int i = 0; i < 8; i++) buf[i] = 777;
            ramp(buf, 0, 1, 1);
            ramp(buf, -3, 1, 1);
            buf[0].Should().Be(777);
            ramp(buf, 1, long.MaxValue, 1);
            buf[0].Should().Be(long.MaxValue);
        }
    }
}
