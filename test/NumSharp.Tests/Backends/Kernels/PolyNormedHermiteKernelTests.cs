using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     The normalised Hermite recurrence kernels behind <c>hermgauss</c> / <c>hermegauss</c>
    ///     (<c>DirectILKernelGenerator.PolyNormedHermite.cs</c>): NumPy 2.4.2's <c>_normed_hermite_n</c> /
    ///     <c>_normed_hermite_e_n</c> bit for bit — NaN payloads and signs included — at special points, against NumPy's
    ///     statements composed from the house ufuncs over every vector / block / tail split, in place, and against
    ///     inaccessible pages.
    /// </summary>
    [TestClass]
    public unsafe class PolyNormedHermiteKernelTests
    {
        // NumPy 2.4.2 (scratchpad hg_kernel_pins.py): fn(POINTS.copy(), n).tobytes().hex() for
        //   POINTS = [nan(0x123), 1.5, -nan(0x456), inf, -0.0, 0.0, -inf, 5e-324, 1e300, -2.25, nan(0xabcd)],
        //   fn = numpy.polynomial.hermite._normed_hermite_n / hermite_e._normed_hermite_e_n.
        /// <summary>NumPy's recurrence bytes at the special points (11 of them: a 4-lane vector, the 2-vector block and
        ///     the scalar tail each see special lanes).</summary>
        private static readonly (PolyNormedHermite Kind, int N, string X, string Result)[] NumPyPins =
        {
            (PolyNormedHermite.Hermite, 0, Points, "105915703809e83f105915703809e83f105915703809e83f105915703809e83f105915703809e83f105915703809e83f105915703809e83f105915703809e83f105915703809e83f105915703809e83f105915703809e83f"),
            (PolyNormedHermite.Hermite, 1, Points, "230100000000f87f2f12a4d2797ef93f560400000000f8ff000000000000f07f00000000000000000000000000000000000000000000f0ff0100000000000000f66581e6fb60397ea30dfb5ddb1e03c0cdab00000000f87f"),
            (PolyNormedHermite.Hermite, 2, Points, "230100000000f87fe1bf14cb38befd3f560400000000f8ff000000000000f07fcab6c2e1fbfee0bfcab6c2e1fbfee0bf000000000000f07fcab6c2e1fbfee0bf000000000000f07f7e18824dd7621340cdab00000000f87f"),
            (PolyNormedHermite.Hermite, 3, Points, "230100000000f87ffdbdd9914439ef3f560400000000f8ff000000000000f8ff00000000000000000000000000000080000000000000f8ff0200000000000080000000000000f07f36ede51101cf1bc0cdab00000000f87f"),
            (PolyNormedHermite.Hermite, 4, Points, "230100000000f87f0f2b3e081466e2bf560400000000f8ff000000000000f8ffe444300d2070dd3fe444300d2070dd3f000000000000f8ffe444300d2070dd3f000000000000f8ff3ec44ce451741b40cdab00000000f87f"),
            (PolyNormedHermite.Hermite, 10, Points, "230100000000f87fe46003c343d6f0bf560400000000f8ff000000000000f8ff4067ac8bf5d8d7bf4067ac8bf5d8d7bf000000000000f8ff4067ac8bf5d8d7bf000000000000f8ffb8e6d5c628f11140cdab00000000f87f"),
            (PolyNormedHermite.Hermite, 40, Points, "230100000000f87f444d8749b6fee03f560400000000f8ff000000000000f8ffc08342396e05d13fc08342396e05d13f000000000000f8ffc08342396e05d13f000000000000f8ff62d52a664849f43fcdab00000000f87f"),
            (PolyNormedHermite.HermiteE, 0, Points, "d2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43fd2ae3e953836e43f"),
            (PolyNormedHermite.HermiteE, 1, Points, "230100000000f87f3b06dedf5451ee3f560400000000f8ff000000000000f07f00000000000000000000000000000000000000000000f0ff0100000000000000ba7dfa73422e2e7eac84e6a7ffbcf6bfcdab00000000f87f"),
            (PolyNormedHermite.HermiteE, 2, Points, "230100000000f87fc65c789168dde13f560400000000f8ff000000000000f07fa3c7f31b7495dcbfa3c7f31b7495dcbf000000000000f07fa3c7f31b7495dcbf000000000000f07fc39663ecc907fd3fcdab00000000f87f"),
            (PolyNormedHermite.HermiteE, 3, Points, "230100000000f87f174ff753d390d2bf560400000000f8ff000000000000f8ff00000000000000000000000000000080000000000000f8ff0200000000000080000000000000f07f8e0997ee5925f3bfcdab00000000f87f"),
            (PolyNormedHermite.HermiteE, 4, Points, "230100000000f87fe5d4ca5aff6ee6bf560400000000f8ff000000000000f8ffc7be49c519c1d83fc7be49c519c1d83f000000000000f8ffc7be49c519c1d83f000000000000f8ff08d8a7ffc7d0ccbfcdab00000000f87f"),
            (PolyNormedHermite.HermiteE, 10, Points, "230100000000f87f1824bbbbf195aebf560400000000f8ff000000000000f8ffc8c9fd65a30dd4bfc8c9fd65a30dd4bf000000000000f8ffc8c9fd65a30dd4bf000000000000f8ff63af673465dae7bfcdab00000000f87f"),
            (PolyNormedHermite.HermiteE, 40, Points, "230100000000f87fce9526c77215d9bf560400000000f8ff000000000000f8ff504166a44ba0cc3f504166a44ba0cc3f000000000000f8ff504166a44ba0cc3f000000000000f8ff0093bbe480edb5bfcdab00000000f87f"),
        };

        /// <summary>The special points' bytes (see <see cref="NumPyPins"/>).</summary>
        private const string Points =
            "230100000000f87f000000000000f83f560400000000f8ff000000000000f07f00000000000000800000000000000000000000000000f0ff01000000000000009c7500883ce4377e00000000000002c0cdab00000000f87f";

        /// <summary>
        ///     The kernel at NumPy's special points gives NumPy's bytes: a NaN point keeps its payload and sign (the second
        ///     operand of <c>c1 * x</c> and of <c>tmp + P</c> wins), an infinite point turns into the default NaN of
        ///     <c>inf - inf</c> where the recurrence meets one, <c>0. + (-0.0)</c> is <c>+0.0</c>, and degree 0 is the constant.
        /// </summary>
        [TestMethod]
        public void SpecialPoints_AreNumPysBits()
        {
            foreach (var (kind, n, xHex, want) in NumPyPins)
            {
                double[] x = FromHex(xHex);
                Assert.AreEqual(want, ToHex(Run(kind, x, n)), $"{kind} n={n}");
            }
        }

        /// <summary>
        ///     Over every count around the vector, block and tail splits and degrees 1-24, 40, 64 and 100, with special values
        ///     scattered among ordinary points, the kernel gives the bytes of NumPy's statements composed from the house ufuncs
        ///     (whose NaN operand order is NumPy's, gated by the nan_order oracle tier).
        /// </summary>
        [TestMethod]
        public void EveryCountAndDegree_IsTheStatementsComposed()
        {
            var rng = new System.Random(20261018);
            // Every split of the cascade (blocks of 4 vectors, at most one block of 2, single vectors, the scalar tail) at 4
            // and at 8 lanes (V256 / V512), alone and combined: e.g. 47 = 2 blocks + the 2-vector block + 1 vector + 3
            // scalar points at 4 lanes.
            int[] counts = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 23, 24, 25, 28, 31, 32, 33, 39, 40, 47, 48, 55,
                             63, 64, 65 };
            int[] degrees = new int[27];
            for (int i = 0; i < 24; i++)
                degrees[i] = i + 1;
            degrees[24] = 40;
            degrees[25] = 64;
            degrees[26] = 100;
            foreach (PolyNormedHermite kind in new[] { PolyNormedHermite.Hermite, PolyNormedHermite.HermiteE })
            foreach (int n in degrees)
            foreach (int count in counts)
            {
                double[] x = RandomPoints(rng, count, n);
                string label = $"{kind} n={n} count={count}";
                Assert.AreEqual(ToHex(Reference(kind, x, n)), ToHex(Run(kind, x, n)), label);
            }
        }

        /// <summary>The destination may be the points themselves: each point is read before its own slot is written.</summary>
        [TestMethod]
        public void InPlace_GivesTheSameBytes()
        {
            var rng = new System.Random(7);
            foreach (PolyNormedHermite kind in new[] { PolyNormedHermite.Hermite, PolyNormedHermite.HermiteE })
            foreach (int count in new[] { 1, 3, 4, 9, 17, 30 })
            {
                double[] x = RandomPoints(rng, count, 12);
                string want = ToHex(Run(kind, x, 12));
                double[] table = Table(kind, 12);
                fixed (double* px = x)
                fixed (double* pt = table)
                    DirectILKernelGenerator.GetPolyNormedHermiteKernel(kind)(px, px, count, pt, 11);
                Assert.AreEqual(want, ToHex(x), $"{kind} count={count}");
            }
        }

        /// <summary>
        ///     Points, results and table placed against inaccessible pages (below and above): the kernel reads and writes only
        ///     its own elements, on every split (an access outside them ends the test host).
        /// </summary>
        [TestMethod]
        public void BetweenInaccessiblePages_TouchesOnlyItsElements()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");
            var rng = new System.Random(11);
            using var xFence = new FencedMemory(pages: 1);
            using var dFence = new FencedMemory(pages: 1);
            using var tFence = new FencedMemory(pages: 1);
            foreach (PolyNormedHermite kind in new[] { PolyNormedHermite.Hermite, PolyNormedHermite.HermiteE })
            foreach (int count in new[] { 0, 1, 3, 4, 5, 8, 9, 12, 13, 16, 17, 31, 47, 63 })
            foreach (int n in new[] { 1, 2, 3, 7 })
            foreach (bool atEnd in new[] { false, true })
            {
                double[] x = RandomPoints(rng, count, n);
                string want = ToHex(Run(kind, x, n));
                double* px = (double*)xFence.Place(count * sizeof(double), atEnd);
                double* pd = (double*)dFence.Place(count * sizeof(double), !atEnd);
                double* pt = (double*)tFence.Place(2 * (n - 1) * sizeof(double), atEnd);
                for (int i = 0; i < count; i++)
                    px[i] = x[i];
                DirectILKernelGenerator.GetPolyNormedHermiteStepsKernel(kind)(pt, n - 1, n);
                DirectILKernelGenerator.GetPolyNormedHermiteKernel(kind)(px, pd, count, pt, n - 1);
                var got = new double[count];
                for (int i = 0; i < count; i++)
                    got[i] = pd[i];
                Assert.AreEqual(want, ToHex(got), $"{kind} n={n} count={count} atEnd={atEnd}");
            }
        }

        /// <summary>
        ///     The step table holds NumPy's factors <c>np.sqrt((nd - 1.) / nd)</c> and <c>np.sqrt(2. / nd)</c> (hermite_e:
        ///     <c>1. / nd</c>) for <c>nd = n, n - 1, ..., 2</c>, and the degree-(n - 1) table is its suffix from row 1 (which is
        ///     what lets hermgauss write one table for its three recurrences).
        /// </summary>
        [TestMethod]
        public void StepTable_IsNumPysFactors()
        {
            foreach (PolyNormedHermite kind in new[] { PolyNormedHermite.Hermite, PolyNormedHermite.HermiteE })
            {
                double numerator = kind == PolyNormedHermite.Hermite ? 2.0 : 1.0;
                double[] t = Table(kind, 50);
                double nd = 50;
                for (int j = 0; j < 49; j++, nd -= 1.0)
                {
                    Assert.AreEqual(BitConverter.DoubleToInt64Bits(System.Math.Sqrt((nd - 1.0) / nd)), BitConverter.DoubleToInt64Bits(t[2 * j]), $"{kind} s1[{j}]");
                    Assert.AreEqual(BitConverter.DoubleToInt64Bits(System.Math.Sqrt(numerator / nd)), BitConverter.DoubleToInt64Bits(t[2 * j + 1]), $"{kind} s2[{j}]");
                }
                double[] shorter = Table(kind, 49);
                for (int i = 0; i < shorter.Length; i++)
                    Assert.AreEqual(BitConverter.DoubleToInt64Bits(t[i + 2]), BitConverter.DoubleToInt64Bits(shorter[i]), $"{kind} suffix [{i}]");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The recurrence of degree <paramref name="n"/> at <paramref name="x"/> through the kernels (degree 0: NumPy's
        ///     <c>np.full</c> branch, the constant).
        /// </summary>
        /// <param name="kind">The recurrence.</param>
        /// <param name="x">The points.</param>
        /// <param name="n">The degree (0 or more).</param>
        /// <returns>The values.</returns>
        private static double[] Run(PolyNormedHermite kind, double[] x, int n)
        {
            var r = new double[x.Length];
            if (n == 0)
            {
                Array.Fill(r, DirectILKernelGenerator.PolyNormedHermiteC1(kind));
                return r;
            }
            double[] table = Table(kind, n);
            fixed (double* px = x)
            fixed (double* pr = r)
            fixed (double* pt = table)
                DirectILKernelGenerator.GetPolyNormedHermiteKernel(kind)(px, pr, x.Length, pt, n - 1);
            return r;
        }

        /// <summary>The step table of degree <paramref name="n"/> (at least one double, so it can be pinned).</summary>
        /// <param name="kind">The recurrence.</param>
        /// <param name="n">The degree (1 or more).</param>
        /// <returns>The table.</returns>
        private static double[] Table(PolyNormedHermite kind, int n)
        {
            var t = new double[System.Math.Max(2 * (n - 1), 1)];
            fixed (double* pt = t)
                DirectILKernelGenerator.GetPolyNormedHermiteStepsKernel(kind)(pt, n - 1, n);
            return t;
        }

        /// <summary>
        ///     NumPy's statements (see the kernel's file header) composed from the house ufuncs, scalars as 0-d arrays, as
        ///     NumPy's np.float64 scalars are: the reference of every kernel run.
        /// </summary>
        /// <param name="kind">The recurrence.</param>
        /// <param name="points">The points.</param>
        /// <param name="n">The degree (1 or more).</param>
        /// <returns>The values.</returns>
        /// <exception cref="InvalidOperationException">The composed result does not have one element per point (a broken
        ///     reference, not a kernel defect).</exception>
        private static double[] Reference(PolyNormedHermite kind, double[] points, int n)
        {
            bool physicists = kind == PolyNormedHermite.Hermite;
            NDArray x = np.array(points);
            NDArray c0 = NDArray.Scalar(0.0);
            NDArray c1 = NDArray.Scalar(1.0 / System.Math.Sqrt(System.Math.Sqrt(physicists ? System.Math.PI : 2 * System.Math.PI)));
            double nd = n;
            for (int i = 0; i < n - 1; i++)
            {
                NDArray tmp = c0;
                c0 = np.multiply(np.negative(c1), NDArray.Scalar(System.Math.Sqrt((nd - 1.0) / nd)));
                c1 = np.add(tmp, np.multiply(np.multiply(c1, x), NDArray.Scalar(System.Math.Sqrt((physicists ? 2.0 : 1.0) / nd))));
                nd -= 1.0;
            }
            NDArray p = np.multiply(c1, x);
            if (physicists)
                p = np.multiply(p, NDArray.Scalar(System.Math.Sqrt(2.0)));
            NDArray r = np.add(c0, p);
            // A count of 0 broadcasts to (0,); degree 1 with one point stays (1,): the elements are what is compared.
            return r.size == points.Length ? r.ToArray<double>() : throw new InvalidOperationException($"reference shape {r.Shape}");
        }

        /// <summary>
        ///     <paramref name="count"/> points mostly within the recurrence's oscillating range for degree <paramref name="n"/>,
        ///     with about one in six a special value: a NaN with a payload (either sign), ±inf, ±0, a subnormal, or a value
        ///     so large the recurrence overflows.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="count">The point count.</param>
        /// <param name="n">The degree (scales the ordinary range, about the largest root's sqrt(2n + 1)).</param>
        /// <returns>The points.</returns>
        private static double[] RandomPoints(System.Random rng, int count, int n)
        {
            var x = new double[count];
            double range = 1.2 * System.Math.Sqrt(2 * n + 1);
            for (int i = 0; i < count; i++)
            {
                x[i] = rng.Next(6) == 0
                    ? rng.Next(8) switch
                    {
                        0 => BitConverter.Int64BitsToDouble(0x7FF8000000000000 | rng.Next(1, 1 << 20)),
                        1 => BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL) | rng.Next(1, 1 << 20)),
                        2 => double.PositiveInfinity,
                        3 => double.NegativeInfinity,
                        4 => -0.0,
                        5 => 0.0,
                        6 => double.Epsilon * rng.Next(1, 1000),
                        _ => (rng.NextDouble() - 0.5) * 1e300,
                    }
                    : (2 * rng.NextDouble() - 1) * range;
            }
            return x;
        }

        /// <summary>The doubles of NumPy's <c>tobytes().hex()</c>.</summary>
        /// <param name="hex">The hex text.</param>
        /// <returns>The values.</returns>
        private static double[] FromHex(string hex)
        {
            byte[] b = Convert.FromHexString(hex);
            var d = new double[b.Length / 8];
            Buffer.BlockCopy(b, 0, d, 0, b.Length);
            return d;
        }

        /// <summary>The bytes of <paramref name="d"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="d">The values.</param>
        /// <returns>The hex text.</returns>
        private static string ToHex(double[] d)
        {
            var b = new byte[d.Length * 8];
            Buffer.BlockCopy(d, 0, b, 0, b.Length);
            return Convert.ToHexString(b).ToLowerInvariant();
        }
    }
}
