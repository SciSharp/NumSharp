using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    ///     <c>-0.0</c> and <c>+0.0</c> are EQUAL to NumPy's sort comparators, so its stable value sort keeps the zeros in
    ///     input order: <c>np.sort([0.0, -0.0, 1.0, -0.0, 0.0], kind='stable')</c> is <c>[0., -0., -0., 0., 1.]</c> (parity
    ///     QA-26). NumSharp's float32 / float64 value sort radix-sorted the raw-bit keys and put every <c>-0.0</c> before every
    ///     <c>+0.0</c>; it now gives the stable order for every kind (NumPy's default kind is CPU-dependent: on an AVX2 host it
    ///     is not even a permutation of the zero signs, so the stable order is the deterministic target). Sign bits are
    ///     compared raw (<c>0.0 == -0.0</c>, so a value assertion proves nothing here).
    /// </summary>
    [TestClass]
    public class NpSortSignedZeroTests
    {
        /// <summary>The sign of each element of a float64 result, as <c>'-'</c> / <c>'+'</c>, read from its raw bits.</summary>
        /// <param name="a">A float64 array.</param>
        /// <returns>One character per element.</returns>
        private static string Signs64(NDArray a)
        {
            var v = a.ToArray<double>();
            var chars = new char[v.Length];
            for (int i = 0; i < v.Length; i++)
                chars[i] = BitConverter.DoubleToInt64Bits(v[i]) < 0 ? '-' : '+';
            return new string(chars);
        }

        /// <summary>The sign of each element of a float32 result, as <c>'-'</c> / <c>'+'</c>, read from its raw bits.</summary>
        /// <param name="a">A float32 array.</param>
        /// <returns>One character per element.</returns>
        private static string Signs32(NDArray a)
        {
            var v = a.ToArray<float>();
            var chars = new char[v.Length];
            for (int i = 0; i < v.Length; i++)
                chars[i] = BitConverter.SingleToInt32Bits(v[i]) < 0 ? '-' : '+';
            return new string(chars);
        }

        /// <summary>
        ///     NumPy 2.4.2: the stable sort of <c>[0.0, -0.0, 1.0, -0.0, 0.0]</c> is <c>[0., -0., -0., 0., 1.]</c> for both float
        ///     widths and every kind here, and <c>np.sort([0.0, -0.0])</c> (the default kind) leaves the pair as it is.
        /// </summary>
        [TestMethod]
        public void SignedZeros_KeepInputOrder_BothWidths_EveryKind()
        {
            using var a = np.array(new[] { 0.0, -0.0, 1.0, -0.0, 0.0 });
            using var stable = np.sort(a, kind: "stable");
            Signs64(stable).Should().Be("+--++");
            stable.ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 0.0, 1.0);
            using var quick = np.sort(a);
            Signs64(quick).Should().Be("+--++");

            using var a32 = np.array(new[] { 0.0f, -0.0f, 1.0f, -0.0f, 0.0f });
            using var stable32 = np.sort(a32, kind: "stable");
            Signs32(stable32).Should().Be("+--++");

            using var pair = np.array(new[] { 0.0, -0.0 });
            using var sortedPair = np.sort(pair);
            Signs64(sortedPair).Should().Be("+-");
        }

        /// <summary>
        ///     The zero run sits between the negatives and the positives with each zero's own sign in input order, NaNs last
        ///     with their bits: the stable kind in input order, the default kind in x86-simd-sort's order (NumPy 2.5; 2.4
        ///     wrote the positive quiet NaN). NumPy's default kind writes this zero run as three <c>-0.0</c> on an AVX2 host,
        ///     not even a permutation of the signs — the input order is NumSharp's deterministic answer for it.
        /// </summary>
        [TestMethod]
        public void ZeroRun_SitsBetweenNegativesAndPositives_NaNsStayLast()
        {
            double negNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000001UL));
            using var a = np.array(new[] { 2.0, -0.0, negNaN, -3.0, 0.0, -0.0, 1.0 });

            using var stable = np.sort(a, kind: "stable");
            var s = stable.ToArray<double>();
            s[0].Should().Be(-3.0);
            Signs64(stable).Substring(1, 3).Should().Be("-+-", "the zeros of -0.0, 0.0, -0.0 in input order");
            s[4].Should().Be(1.0);
            s[5].Should().Be(2.0);
            BitConverter.DoubleToInt64Bits(s[6]).Should().Be(unchecked((long)0xFFF8000000000001UL), "the stable kind keeps the NaN's bits");

            using var quick = np.sort(a);
            var q = quick.ToArray<double>();
            Signs64(quick).Substring(1, 3).Should().Be("-+-");
            BitConverter.DoubleToInt64Bits(q[6]).Should().Be(unchecked((long)0xFFF8000000000001UL), "the default kind keeps the NaN's bits too");
        }

        /// <summary>
        ///     An axis sort keeps each line's zeros in that line's input order (strided lines included), an all-zero line keeps
        ///     its signs, and ndarray.sort() in place gives the same answer as np.sort.
        /// </summary>
        [TestMethod]
        public void AxisLines_AndInPlaceSort_KeepEachLinesZeroOrder()
        {
            using var m = np.array(new[,] { { -0.0, 0.0, -1.0 }, { 0.0, -0.0, -0.0 } });
            using var rows = np.sort(m, axis: 1);
            Signs64(rows).Should().Be("--+" + "+--", "row 0: -1, -0, +0; row 1: the all-zero row keeps +0, -0, -0");

            using var cols = np.sort(m, axis: 0);
            Signs64(cols).Should().Be("-+-" + "+--", "column 0: -0, +0; column 1: +0, -0; column 2: -1, -0");

            using var inPlace = np.array(new[] { 0.0f, 5.0f, -0.0f });
            inPlace.sort();
            Signs32(inPlace).Should().Be("+-+");
            inPlace.ToArray<float>()[2].Should().Be(5.0f);
        }
    }
}
