using System;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    ///     NumPy 2.5's sort keywords (gh-31345): <c>descending=True</c> on <c>np.sort</c> / <c>np.argsort</c> /
    ///     <c>ndarray.sort</c> / <c>ndarray.argsort</c>, the heapsort kind as the default kind for every dtype, the
    ///     refusals (StringDType, object arrays, numpy.ma, a kind given with a keyword), and the default kind's float NaN tail
    ///     in the order NumPy 2.5's x86-simd-sort leaves it. Every expected value is NumPy 2.5.4's output (win-amd64, AVX2);
    ///     float results are compared by their raw bits, since <c>0.0 == -0.0</c> and NaN != NaN prove nothing by value.
    /// </summary>
    [TestClass]
    public class NpSortDescendingTests
    {
        /// <summary>The raw bits of a float64 array, element by element.</summary>
        /// <param name="a">A float64 array.</param>
        /// <returns>The bits.</returns>
        private static ulong[] Bits64(NDArray a)
        {
            var v = a.ToArray<double>();
            var bits = new ulong[v.Length];
            for (int i = 0; i < v.Length; i++)
                bits[i] = BitConverter.DoubleToUInt64Bits(v[i]);
            return bits;
        }

        /// <summary>The raw bits of a float32 array, element by element.</summary>
        /// <param name="a">A float32 array.</param>
        /// <returns>The bits.</returns>
        private static uint[] Bits32(NDArray a)
        {
            var v = a.ToArray<float>();
            var bits = new uint[v.Length];
            for (int i = 0; i < v.Length; i++)
                bits[i] = BitConverter.SingleToUInt32Bits(v[i]);
            return bits;
        }

        /// <summary>A float64 array built from raw bits (NaN payloads and signs kept).</summary>
        /// <param name="bits">The bits.</param>
        /// <returns>The array.</returns>
        private static NDArray FromBits64(ulong[] bits)
        {
            var v = new double[bits.Length];
            for (int i = 0; i < bits.Length; i++)
                v[i] = BitConverter.UInt64BitsToDouble(bits[i]);
            return np.array(v);
        }

        /// <summary>A float32 array built from raw bits (NaN payloads and signs kept).</summary>
        /// <param name="bits">The bits.</param>
        /// <returns>The array.</returns>
        private static NDArray FromBits32(uint[] bits)
        {
            var v = new float[bits.Length];
            for (int i = 0; i < bits.Length; i++)
                v[i] = BitConverter.UInt32BitsToSingle(bits[i]);
            return np.array(v);
        }

        /// <summary>The float64 line of the NaN-tail and signed-zero cases: <c>[0, -0, 1, -0, 0, nan(payload 1), -nan, -1] * 3</c>
        ///     with a second payload at index 13.</summary>
        private static readonly ulong[] F64Input =
        {
            0x0000000000000000UL, 0x8000000000000000UL, 0x3FF0000000000000UL, 0x8000000000000000UL, 0x0000000000000000UL,
            0x7FF8000000000001UL, 0xFFF8000000000000UL, 0xBFF0000000000000UL, 0x0000000000000000UL, 0x8000000000000000UL,
            0x3FF0000000000000UL, 0x8000000000000000UL, 0x0000000000000000UL, 0xFFF8000000000002UL, 0xFFF8000000000000UL,
            0xBFF0000000000000UL, 0x0000000000000000UL, 0x8000000000000000UL, 0x3FF0000000000000UL, 0x8000000000000000UL,
            0x0000000000000000UL, 0x7FF8000000000000UL, 0xFFF8000000000000UL, 0xBFF0000000000000UL,
        };

        /// <summary>
        ///     The keywords: <c>np.sort([3, nan, 1, 2, 3], descending=True)</c> is <c>[3, 3, 2, 1, nan]</c> (NaN still last),
        ///     its argsort <c>[0, 4, 3, 2, 1]</c> for both kinds; <c>descending=False</c> / null sort ascending; the ndarray
        ///     methods take the keyword too.
        /// </summary>
        [TestMethod]
        public void Descending_LargestFirst_NaNLast()
        {
            using var v = np.array(new[] { 3.0, double.NaN, 1.0, 2.0, 3.0 });
            using var sorted = np.sort(v, descending: true);
            sorted.ToArray<double>().Should().Equal(3.0, 3.0, 2.0, 1.0, double.NaN);
            using var order = np.argsort(v, descending: true);
            order.ToArray<long>().Should().Equal(0L, 4, 3, 2, 1);
            using var stableOrder = np.argsort(v, stable: true, descending: true);
            stableOrder.ToArray<long>().Should().Equal(0L, 4, 3, 2, 1);
            using var ascending = np.sort(v, descending: false);
            ascending.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 3.0, double.NaN);

            using var copy = v.copy();
            copy.sort(descending: true);
            copy.ToArray<double>().Should().Equal(3.0, 3.0, 2.0, 1.0, double.NaN);
            using var methodOrder = v.argsort(descending: true);
            methodOrder.ToArray<long>().Should().Equal(0L, 4, 3, 2, 1);
        }

        /// <summary>
        ///     The default kind's tie order: NumPy runs its scalar introsort for a descending sort (x86-simd-sort is
        ///     ascending-only), so 40 elements of four values come back in that introsort's order, not the stable one;
        ///     the stable kind keeps ties in input order.
        /// </summary>
        [TestMethod]
        public void Descending_DefaultKind_IsNumPysIntrosortTieOrder()
        {
            using var ties = np.array(new long[]
            {
                2, 3, 0, 3, 1, 2, 2, 1, 3, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 3, 0, 2, 3, 0, 1, 1, 1, 3, 0, 3, 3, 3, 0, 1, 2, 1, 2, 2, 2, 0,
            });
            using var quick = np.argsort(ties, descending: true);
            quick.ToArray<long>().Should().Equal(
                19L, 1, 3, 31, 30, 8, 29, 27, 22, 37, 36, 34, 21, 38, 0, 5, 12, 6, 10, 4,
                35, 13, 11, 33, 24, 25, 26, 7, 2, 32, 15, 28, 14, 23, 20, 18, 17, 16, 9, 39);
            using var stable = np.argsort(ties, stable: true, descending: true);
            stable.ToArray<long>().Should().Equal(
                1L, 3, 8, 19, 22, 27, 29, 30, 31, 0, 5, 6, 12, 21, 34, 36, 37, 38, 4, 7,
                10, 11, 13, 24, 25, 26, 33, 35, 2, 9, 14, 15, 16, 17, 18, 20, 23, 28, 32, 39);
        }

        /// <summary>
        ///     float64 descending, bit for bit: the default kind places the zeros and the NaN payloads where NumPy's value
        ///     introsort puts them, the stable kind keeps both in input order; the argsorts agree.
        /// </summary>
        [TestMethod]
        public void Descending_Float64_ZerosAndNaNPayloads_BothKinds()
        {
            using var f = FromBits64(F64Input);
            using var quick = np.sort(f, descending: true);
            Bits64(quick).Should().Equal(
                0x3FF0000000000000UL, 0x3FF0000000000000UL, 0x3FF0000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL,
                0x8000000000000000UL, 0x8000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x8000000000000000UL,
                0x0000000000000000UL, 0x0000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL,
                0xBFF0000000000000UL, 0xBFF0000000000000UL, 0xBFF0000000000000UL, 0xFFF8000000000000UL, 0xFFF8000000000002UL,
                0xFFF8000000000000UL, 0xFFF8000000000000UL, 0x7FF8000000000001UL, 0x7FF8000000000000UL);
            using var stable = np.sort(f, stable: true, descending: true);
            Bits64(stable).Should().Equal(
                0x3FF0000000000000UL, 0x3FF0000000000000UL, 0x3FF0000000000000UL, 0x0000000000000000UL, 0x8000000000000000UL,
                0x8000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL,
                0x0000000000000000UL, 0x0000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x0000000000000000UL,
                0xBFF0000000000000UL, 0xBFF0000000000000UL, 0xBFF0000000000000UL, 0x7FF8000000000001UL, 0xFFF8000000000000UL,
                0xFFF8000000000002UL, 0xFFF8000000000000UL, 0x7FF8000000000000UL, 0xFFF8000000000000UL);
            using var quickOrder = np.argsort(f, descending: true);
            quickOrder.ToArray<long>().Should().Equal(2L, 18, 10, 0, 20, 19, 17, 16, 12, 11, 8, 4, 3, 1, 9, 7, 15, 23, 22, 13, 14, 6, 5, 21);
            using var stableOrder = np.argsort(f, stable: true, descending: true);
            stableOrder.ToArray<long>().Should().Equal(2L, 10, 18, 0, 1, 3, 4, 8, 9, 11, 12, 16, 17, 19, 20, 7, 15, 23, 5, 6, 13, 14, 21, 22);
        }

        /// <summary>
        ///     The ASCENDING default kind keeps every NaN's bits since NumPy 2.5.2 (x86-simd-sort fa944ef; NumPy 2.4 wrote the
        ///     positive quiet NaN), in the order its <c>move_nans_to_end_of_array</c> swap pass leaves them; the stable kind
        ///     keeps them in input order. (The zeros' order of the default kind is the SIMD sort's, which NumSharp does not
        ///     reproduce: only the tail is asserted.)
        /// </summary>
        [TestMethod]
        public void Ascending_DefaultKind_NaNTailIsX86SimdSortsSwapOrder()
        {
            using var f = FromBits64(F64Input);
            using var quick = np.sort(f);
            Bits64(quick)[18..].Should().Equal(
                0xFFF8000000000000UL, 0xFFF8000000000002UL, 0x7FF8000000000000UL, 0xFFF8000000000000UL, 0xFFF8000000000000UL,
                0x7FF8000000000001UL);
            using var stable = np.sort(f, kind: "stable");
            Bits64(stable)[18..].Should().Equal(
                0x7FF8000000000001UL, 0xFFF8000000000000UL, 0xFFF8000000000002UL, 0xFFF8000000000000UL, 0x7FF8000000000000UL,
                0xFFF8000000000000UL);

            var f32Bits = new uint[]
            {
                0x00000000u, 0x80000000u, 0x3F800000u, 0x80000000u, 0x00000000u, 0x7FC00001u, 0xFFC00000u, 0xBF800000u,
                0x00000000u, 0x80000000u, 0x3F800000u, 0x80000000u, 0x00000000u, 0xFFC00000u, 0xFFC00000u, 0xBF800000u,
                0x00000000u, 0x80000000u, 0x3F800000u, 0x80000000u, 0x00000000u, 0x7FC00000u, 0xFFC00000u, 0xBF800000u,
            };
            using var g = FromBits32(f32Bits);
            using var quick32 = np.sort(g);
            Bits32(quick32)[18..].Should().Equal(0xFFC00000u, 0xFFC00000u, 0x7FC00000u, 0xFFC00000u, 0xFFC00000u, 0x7FC00001u);
            using var desc32 = np.sort(g, descending: true);
            Bits32(desc32).Should().Equal(
                0x3F800000u, 0x3F800000u, 0x3F800000u, 0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u, 0x00000000u,
                0x00000000u, 0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u, 0xBF800000u,
                0xBF800000u, 0xBF800000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0x7FC00001u, 0x7FC00000u);
        }

        /// <summary>
        ///     complex128 keeps its NaN classes last in the ascending order (no NaN part, R+nanj, nan+Rj, nan+nanj), each
        ///     class descending; float16 keeps NaN last, the zeros equal; bool, uint64 and int16 along either axis and flattened.
        /// </summary>
        [TestMethod]
        public void Descending_ComplexHalfIntegers()
        {
            var cx = new System.Numerics.Complex[20];
            var pattern = new[]
            {
                new System.Numerics.Complex(1, 2), new System.Numerics.Complex(double.NaN, 1), new System.Numerics.Complex(1, 2),
                new System.Numerics.Complex(1, double.NaN), new System.Numerics.Complex(double.NaN, double.NaN),
                new System.Numerics.Complex(0, 0), new System.Numerics.Complex(-0.0, 0), new System.Numerics.Complex(2, -1),
                new System.Numerics.Complex(double.NaN, 3), new System.Numerics.Complex(2, double.NaN),
            };
            for (int i = 0; i < 20; i++)
                cx[i] = pattern[i % 10];
            using var c = np.array(cx);
            using var cQuick = np.argsort(c, descending: true);
            cQuick.ToArray<long>().Should().Equal(17L, 7, 0, 2, 12, 10, 16, 15, 5, 6, 9, 19, 13, 3, 8, 18, 11, 1, 14, 4);
            using var cStable = np.argsort(c, stable: true, descending: true);
            cStable.ToArray<long>().Should().Equal(7L, 17, 0, 2, 10, 12, 5, 6, 15, 16, 9, 19, 3, 13, 8, 18, 1, 11, 4, 14);

            // NumPy's np.nan is the POSITIVE quiet NaN (.NET's double.NaN is negative): float16 0x7E00.
            double nan = BitConverter.UInt64BitsToDouble(0x7FF8000000000000UL);
            using var h = np.array(new[] { 1.0, nan, -2.0, 0.0, -0.0, 3.0, double.PositiveInfinity, double.NegativeInfinity })
                .astype(NPTypeCode.Half);
            using var hSorted = np.sort(h, descending: true);
            var hBits = Array.ConvertAll(hSorted.ToArray<Half>(), x => (long)BitConverter.HalfToUInt16Bits(x));
            hBits.Should().Equal(31744L, 16896, 15360, 0, 32768, 49152, 64512, 32256);
            using var hOrder = np.argsort(h, descending: true);
            hOrder.ToArray<long>().Should().Equal(6L, 5, 0, 3, 4, 2, 7, 1);

            using var b = np.array(new[] { true, false, true, false });
            using var bSorted = np.sort(b, descending: true);
            bSorted.ToArray<bool>().Should().Equal(true, true, false, false);
            using var bOrder = np.argsort(b, descending: true);
            bOrder.ToArray<long>().Should().Equal(0L, 2, 1, 3);
            using var u = np.array(new ulong[] { 5, ulong.MaxValue, 0 });
            using var uSorted = np.sort(u, descending: true);
            uSorted.ToArray<ulong>().Should().Equal(ulong.MaxValue, 5UL, 0UL);

            using var m = np.array(new short[,] { { 3, 1, 2 }, { 9, 9, 0 } });
            using var axis0 = np.argsort(m, 0, descending: true);
            axis0.ToArray<long>().Should().Equal(1L, 1, 0, 0, 0, 1);
            using var axis0Values = np.sort(m, 0, descending: true);
            axis0Values.ToArray<short>().Should().Equal((short)9, 9, 2, 3, 1, 0);
            using var flat = np.argsort(m, null, descending: true);
            flat.ToArray<long>().Should().Equal(3L, 4, 0, 2, 1, 5);
            using var flatValues = np.sort(m, null, descending: true);
            flatValues.ToArray<short>().Should().Equal((short)9, 9, 3, 2, 1, 0);
        }

        /// <summary>
        ///     Fixed-width text: <c>string_aquicksort_</c> with reverse for the default kind (NumPy's tie order over 40 strings of
        ///     three values), <c>string_atimsort_</c> for the stable kind, and the value sort largest first.
        /// </summary>
        [TestMethod]
        public void Descending_Text_BothKinds()
        {
            var values = new string[40];
            var pattern = new[] { "b", "a", "c", "a", "b" };
            for (int i = 0; i < 40; i++)
                values[i] = pattern[i % 5];
            using var u = np.array(values);
            using var quick = np.argsort(u, descending: true);
            quick.ToArray<long>().Should().Equal(
                37L, 2, 32, 7, 27, 12, 22, 17, 0, 35, 34, 30, 29, 25, 24, 20, 19, 15, 4, 5,
                9, 10, 39, 14, 23, 1, 36, 3, 18, 33, 38, 31, 6, 8, 16, 26, 21, 11, 13, 28);
            using var stable = np.argsort(u, stable: true, descending: true);
            stable.ToArray<long>().Should().Equal(
                2L, 7, 12, 17, 22, 27, 32, 37, 0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29,
                30, 34, 35, 39, 1, 3, 6, 8, 11, 13, 16, 18, 21, 23, 26, 28, 31, 33, 36, 38);
            using var sorted = np.sort(u, descending: true);
            var expected = new string[40];
            for (int i = 0; i < 40; i++)
                expected[i] = i < 8 ? "c" : i < 24 ? "b" : "a";
            NDArray.AsStringArray(sorted).Should().Equal(expected);
            using var flat = np.argsort(u.reshape(5, 8), null, descending: true);
            flat.ToArray<long>().Should().Equal(quick.ToArray<long>());
        }

        /// <summary>
        ///     A kind given together with either keyword is refused before the array is read — <c>descending=False</c> and
        ///     <c>stable=False</c> included (NumPy reads any non-None value as given).
        /// </summary>
        [TestMethod]
        public void KindWithKeywords_IsRefused()
        {
            using var v = np.array(new[] { 3.0, 1.0 });
            const string message = "`kind` and keyword parameters can't be provided at the same time. Use only one of them.";
            Action sortBoth = () => np.sort(v, kind: "stable", descending: true);
            sortBoth.Should().Throw<ValueError>().WithMessage(message);
            Action sortFalse = () => np.sort(v, kind: "stable", descending: false);
            sortFalse.Should().Throw<ValueError>().WithMessage(message);
            Action argsortBoth = () => np.argsort(v, kind: "quicksort", descending: true);
            argsortBoth.Should().Throw<ValueError>().WithMessage(message);
            Action method = () => v.sort(kind: "heapsort", descending: false);
            method.Should().Throw<ValueError>().WithMessage(message);
        }

        /// <summary>
        ///     StringDType: the heapsort kind is the default kind now (NumPy 2.4 refused it); the descending flag is the
        ///     sort loop's <c>RuntimeError</c> — after the axis and writeability checks for a sort, before the axis check
        ///     for an argsort, and for any size.
        /// </summary>
        [TestMethod]
        public void StringDType_HeapsortSorts_DescendingIsRefused()
        {
            using var t = np.array(new[] { "b", "a" }).astype(np.dtype("T"));
            using var heap = np.sort(t, kind: "heapsort");
            NDArray.AsStringArray(heap).Should().Equal("a", "b");
            using var heapOrder = np.argsort(t, kind: "heapsort");
            heapOrder.ToArray<long>().Should().Equal(1L, 0);

            Action sort = () => np.sort(t, descending: true);
            sort.Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");
            Action one = () => np.sort(t["0:1"], descending: true);
            one.Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");
            Action stable = () => np.sort(t, stable: true, descending: true);
            stable.Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");
            Action sortAxis = () => np.sort(t, 5, descending: true);
            sortAxis.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
            Action argsortAxis = () => np.argsort(t, 5, descending: true);
            argsortAxis.Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");
            using var readOnly = np.broadcast_to(t, new Shape(2, 2));
            Action ro = () => readOnly.sort(descending: true);
            ro.Should().Throw<ValueError>().WithMessage("sort array is read-only");
            using var ascending = np.sort(t, descending: false);
            NDArray.AsStringArray(ascending).Should().Equal("a", "b");
        }

        /// <summary>
        ///     An object array of lists has no descending sort function: <c>TypeError</c> — after the axis and writeability
        ///     checks for a sort, before the axis check for an argsort; <c>descending=False</c> sorts.
        /// </summary>
        [TestMethod]
        public void ObjectArray_DescendingIsRefused()
        {
            using var lists = np.@char.split(np.array(new[] { "c", "a", "b" }));
            Action sort = () => lists.sort(descending: true);
            sort.Should().Throw<TypeError>().WithMessage("no current sort function meets the requirements");
            Action sortAxis = () => lists.sort(axis: 5, descending: true);
            sortAxis.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
            Action argsort = () => lists.argsort(descending: true);
            argsort.Should().Throw<TypeError>().WithMessage("no current argsort function meets the requirements");
            Action argsortAxis = () => lists.argsort(axis: 5, descending: true);
            argsortAxis.Should().Throw<TypeError>().WithMessage("no current argsort function meets the requirements");
            using var order = lists.argsort(descending: false);
            order.ToArray<long>().Should().Equal(1L, 2, 0);
        }

        /// <summary>
        ///     numpy.ma: a masked operand refuses <c>stable=True</c> and <c>descending=True</c> (MaskedArray.sort /
        ///     argsort, stable first), and takes false; a plain operand of the module functions sorts through
        ///     <c>ndarray.sort</c> / <c>argsort</c>, which honour them.
        /// </summary>
        [TestMethod]
        public void MaskedArray_RefusesStableAndDescending()
        {
            using var data = np.array(new long[] { 3, 1, 2 });
            var ma = np.ma.array(data);
            Action sort = () => np.ma.sort(ma, descending: true);
            sort.Should().Throw<ValueError>().WithMessage("`descending` parameter is not supported for masked arrays.");
            Action stable = () => np.ma.sort(ma, stable: true, descending: true);
            stable.Should().Throw<ValueError>().WithMessage("`stable` parameter is not supported for masked arrays.");
            Action argsort = () => np.ma.argsort(ma, descending: true);
            argsort.Should().Throw<ValueError>().WithMessage("`descending` parameter is not supported for masked arrays.");
            Action method = () => ma.sort(descending: true);
            method.Should().Throw<ValueError>().WithMessage("`descending` parameter is not supported for masked arrays.");
            Action methodArg = () => ma.argsort(descending: true);
            methodArg.Should().Throw<ValueError>().WithMessage("`descending` parameter is not supported for masked arrays.");
            var ok = np.ma.sort(ma, descending: false);
            ok.data.ToArray<long>().Should().Equal(1L, 2, 3);

            using var plainOrder = np.ma.argsort(data, descending: true);
            plainOrder.ToArray<long>().Should().Equal(0L, 2, 1);
            var plainSorted = np.ma.sort(data, descending: true);
            plainSorted.data.ToArray<long>().Should().Equal(3L, 2, 1);
        }
    }
}
