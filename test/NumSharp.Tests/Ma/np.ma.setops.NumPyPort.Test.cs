using System;
using System.Linq;
using AwesomeAssertions;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     numpy.ma's arraysetops (<c>numpy/ma/extras.py</c>: unique, in1d, isin, intersect1d, setxor1d, union1d,
    ///     setdiff1d) as NumPy's own statements. NumPy builds them from <c>np.unique</c>, <c>ma.concatenate</c>, the masked
    ///     sort / argsort (masked slots keyed by the fill value) and the masked <c>==</c> / <c>!=</c> (two masked slots are
    ///     equal, a masked slot is unequal to every value, and every comparison that reads a masked slot is masked), so a
    ///     flag whose sorted neighbour is masked is itself masked. Every expected value is NumPy 2.4.2's output for the same
    ///     call (most inputs are NumPy's own <c>TestArraySetOps</c> cases).
    /// </summary>
    /// <remarks>
    ///     Before the port (2026-10-08) these ran the plain set operation over the unmasked values and appended one masked
    ///     entry by a presence rule: an in1d / isin result never had a mask (all 8 oracle cases failed once their
    ///     <c>[known gap]</c> excuse was removed), NaN and masked slots never shared a sort, and ma.unique had no
    ///     <c>return_index</c> / <c>return_inverse</c>. The zero-excuse oracle tier <c>ma_setops.jsonl</c> (6,971 cases)
    ///     covers the ground these tests pin by example.
    /// </remarks>
    [TestClass]
    public class np_ma_setops_numpy_port_tests
    {
        private static NDMaskedArray L(long[] data, int[] mask = null) =>
            np.ma.array(np.array(data), mask is null ? null : np.array(mask.Select(m => m != 0).ToArray()));

        private static bool[] Mask(NDMaskedArray r) => np.ma.getmaskarray(r).ToArray<bool>();

        private static bool[] Bools(NDMaskedArray r) => r.data.ToArray<bool>();

        /// <summary>
        ///     The defining case: <c>ma.in1d([3, 5], ma.array([3, 9], mask=[0, 1]))</c> is [True, --]. After the sort, 5 sits
        ///     next to the masked 9, and <c>5 == masked</c> is masked (its data False). The old implementation returned
        ///     [True, False] with no mask.
        /// </summary>
        [TestMethod]
        public void In1d_AValueNextToAMaskedSlot_IsMasked()
        {
            var r = np.ma.in1d(L(new long[] { 3, 5 }), L(new long[] { 3, 9 }, new[] { 0, 1 }));
            Bools(r).Should().Equal(true, false);
            Mask(r).Should().Equal(false, true);
            var inv = np.ma.isin(L(new long[] { 3, 5 }), L(new long[] { 3, 9 }, new[] { 0, 1 }), invert: true);
            Bools(inv).Should().Equal(false, true);
            Mask(inv).Should().Equal(false, true);
            // setdiff1d selects by the inverted flags' DATA, the masked one included: [5].
            var d = np.ma.setdiff1d(L(new long[] { 3, 5 }), L(new long[] { 3, 9 }, new[] { 0, 1 }));
            d.data.ToArray<long>().Should().Equal(5L);
            Mask(d).Should().Equal(false);
        }

        /// <summary>NumPy's test_in1d / test_in1d_invert inputs: the data AND the mask NumPy's statements give.</summary>
        [TestMethod]
        public void In1d_NumPysTests()
        {
            var r = np.ma.in1d(L(new long[] { 1, 2, 5, 7, -1 }, new[] { 0, 0, 0, 0, 1 }),
                               L(new long[] { 1, 2, 3, 4, 5, -1 }, new[] { 0, 0, 0, 0, 0, 1 }));
            Bools(r).Should().Equal(true, true, true, false, true);
            Mask(r).Should().Equal(false, false, false, true, true);

            var a = L(new long[] { 5, 5, 2, 1, -1 }, new[] { 0, 0, 0, 0, 1 });
            var b = L(new long[] { 1, 5, -1 }, new[] { 0, 0, 1 });
            r = np.ma.in1d(a, b);
            Bools(r).Should().Equal(true, true, false, true, true);
            Mask(r).Should().Equal(false, false, false, false, true);
            r = np.ma.in1d(a, b, invert: true);
            Bools(r).Should().Equal(false, false, true, false, false);
            Mask(r).Should().Equal(false, false, false, false, true);
        }

        /// <summary>
        ///     NumPy's test_isin: a 2x3x4 element with one masked slot. isin keeps the element's shape, and so does in1d
        ///     (np.unique's inverse has the input's shape in NumPy 2), and the masked element's flag is masked.
        /// </summary>
        [TestMethod]
        public void Isin_KeepsTheElementsShape()
        {
            var mask = new bool[24];
            mask[20] = true;
            var a = np.ma.array(np.arange(24L).reshape(2, 3, 4), np.array(mask).reshape(2, 3, 4));
            var b = L(new long[] { 0, 10, 20, 30, 1, 3, 11, 22, 33 }, new[] { 0, 1, 0, 1, 0, 1, 0, 1, 0 });
            var r = np.ma.isin(a, b);
            r.shape.Should().Equal(2L, 3L, 4L);
            var expected = new bool[24];
            expected[0] = expected[1] = expected[11] = expected[20] = true;
            np.ravel(r.data).ToArray<bool>().Should().Equal(expected);
            np.ravel(np.ma.getmaskarray(r)).ToArray<bool>().Should().Equal(mask);
            np.ma.in1d(a, b).shape.Should().Equal(2L, 3L, 4L);
        }

        /// <summary>
        ///     NumPy's test_unique_onmaskedarray / test_unique_allmasked: the masked slots collapse into one, which keeps the
        ///     FIRST masked element (index 2, data 1), and the indices / inverse follow (the inverse shaped like the input).
        /// </summary>
        [TestMethod]
        public void Unique_ReturnIndexAndInverse()
        {
            var (values, index, inverse) = np.ma.unique(L(new long[] { 1, 1, 1, 2, 2, 3 }, new[] { 0, 0, 1, 0, 1, 0 }),
                                                        return_index: true, return_inverse: true);
            values.data.ToArray<long>().Should().Equal(1L, 2L, 3L, 1L);
            Mask(values).Should().Equal(false, false, false, true);
            index.ToArray<long>().Should().Equal(0L, 3L, 5L, 2L);
            inverse.ToArray<long>().Should().Equal(0L, 0L, 3L, 1L, 3L, 2L);

            var all = np.ma.unique(L(new long[] { 1, 1, 1 }, new[] { 1, 1, 1 }), return_index: true, return_inverse: true);
            all.values.data.ToArray<long>().Should().Equal(1L);
            Mask(all.values).Should().Equal(true);
            all.indices.ToArray<long>().Should().Equal(0L);
            all.inverse_indices.ToArray<long>().Should().Equal(0L, 0L, 0L);

            // The inverse keeps a 2-D input's shape.
            var two = np.ma.unique(np.ma.array(np.array(new long[,] { { 3, 1 }, { 1, 9 } }), np.array(new[,] { { false, false }, { false, true } })),
                                   return_inverse: true);
            two.values.data.ToArray<long>().Should().Equal(1L, 3L, 9L);
            Mask(two.values).Should().Equal(false, false, true);
            two.inverse_indices.shape.Should().Equal(2L, 2L);
            np.ravel(two.inverse_indices).ToArray<long>().Should().Equal(1L, 0L, 0L, 2L);
        }

        /// <summary><c>unique(masked, return_index=True, return_inverse=True)</c>: one masked float64 value, index [0],
        ///     and a 0-d inverse (the masked constant is 0-d).</summary>
        [TestMethod]
        public void Unique_OfTheMaskedConstant()
        {
            var r = np.ma.unique(np.ma.masked, return_index: true, return_inverse: true);
            r.values.dtype.Should().Be(np.float64);
            Mask(r.values).Should().Equal(true);
            r.indices.ToArray<long>().Should().Equal(0L);
            r.inverse_indices.ndim.Should().Be(0);
            r.inverse_indices.GetInt64().Should().Be(0L);
        }

        /// <summary>
        ///     A masked bool slot is keyed by True (the dtype maximum), so it sorts AMONG the unmasked Trues: NumPy's
        ///     <c>ma.unique([True(masked), True, False])</c> is [False, --, True] — not "the unmasked values plus one trailing
        ///     masked entry".
        /// </summary>
        [TestMethod]
        public void Unique_AMaskedKeyTiedWithAnEqualValue_SortsAmongThem()
        {
            var r = np.ma.unique(np.ma.array(np.array(new[] { true, true, false }), np.array(new[] { true, false, false })));
            Bools(r).Should().Equal(false, true, true);
            Mask(r).Should().Equal(false, true, false);
        }

        /// <summary>
        ///     The NaN collapse runs only behind an UNMASKED NaN (<c>np.isnan(aux[-1])</c> of a masked last element is
        ///     <c>masked</c>, which is falsy): with a masked slot last the NaNs stay distinct; without a mask they collapse.
        /// </summary>
        [TestMethod]
        public void Unique_NaNsBeforeAMaskedSlot_AreNotCollapsed()
        {
            var r = np.ma.unique(np.ma.array(np.array(new[] { double.NaN, 1.0, double.NaN, 2.0 }), np.array(new[] { false, false, false, true })));
            var d = r.data.ToArray<double>();
            d.Length.Should().Be(4);
            d[0].Should().Be(1.0);
            double.IsNaN(d[1]).Should().BeTrue();
            double.IsNaN(d[2]).Should().BeTrue();
            Mask(r).Should().Equal(false, false, false, true);

            // The masked last slot holds NaN DATA: still `masked`, still falsy, so the unmasked NaN is not collapsed into
            // it — NumPy keeps [1.0, nan, --].
            var nanLast = np.ma.unique(np.ma.array(np.array(new[] { double.NaN, 1.0, double.NaN }), np.array(new[] { false, false, true })));
            nanLast.size.Should().Be(3);
            Mask(nanLast).Should().Equal(false, false, true);
            double.IsNaN(nanLast.data.GetDouble(1)).Should().BeTrue();

            var plain = np.ma.unique(np.ma.array(np.array(new[] { double.NaN, 1.0, double.NaN, 2.0 })));
            var p = plain.data.ToArray<double>();
            p.Length.Should().Be(3);
            p[0].Should().Be(1.0);
            p[1].Should().Be(2.0);
            double.IsNaN(p[2]).Should().BeTrue();
            ReferenceEquals(np.ma.getmask(plain), np.ma.nomask).Should().BeTrue();
        }

        /// <summary>
        ///     ma.unique sorts its flattened copy IN PLACE through the masked setter, so on a HARD mask the old and sorted
        ///     masks are OR-ed and the masked slots keep their old data: <c>ma.unique(hard [--, 1, 2])</c> is NumPy's
        ///     <c>[--, 2, --]</c>, and the result is hard.
        /// </summary>
        [TestMethod]
        public void Unique_HardMask_KeepsNumPysSortedWrite()
        {
            var h = L(new long[] { 3, 1, 2 }, new[] { 1, 0, 0 }).harden_mask();
            var r = np.ma.unique(h);
            r.data.ToArray<long>().Should().Equal(3L, 2L, 2L);
            Mask(r).Should().Equal(true, false, true);
            r.hardmask.Should().BeTrue();
        }

        /// <summary>NumPy's test_intersect1d / test_setxor1d / test_union1d / test_setdiff1d inputs.</summary>
        [TestMethod]
        public void SetOperations_NumPysTests()
        {
            var i = np.ma.intersect1d(L(new long[] { 1, 3, 3, 3 }, new[] { 0, 0, 0, 1 }), L(new long[] { 3, 1, 1, 1 }, new[] { 0, 0, 0, 1 }));
            i.data.ToArray<long>()[..2].Should().Equal(1L, 3L);
            Mask(i).Should().Equal(false, false, true);

            var x1 = np.ma.setxor1d(L(new long[] { 1, 2, 5, 7, -1 }, new[] { 0, 0, 0, 0, 1 }),
                                    L(new long[] { 1, 2, 3, 4, 5, -1 }, new[] { 0, 0, 0, 0, 0, 1 }));
            x1.data.ToArray<long>().Should().Equal(3L, 4L, 7L);
            Mask(x1).Should().Equal(false, false, false);
            var x2 = np.ma.setxor1d(L(new long[] { 1, 2, 5, 7, -1 }, new[] { 0, 0, 0, 0, 1 }), np.array(new long[] { 1, 2, 3, 4, 5 }));
            x2.data.ToArray<long>()[..3].Should().Equal(3L, 4L, 7L);
            Mask(x2).Should().Equal(false, false, false, true);
            // assume_unique joins the flattened operands (axis=None): [[1],[8],[2],[3]] with [[6,5],[4,8]].
            var x3 = np.ma.setxor1d(np.array(new long[,] { { 1 }, { 8 }, { 2 }, { 3 } }), np.array(new long[,] { { 6, 5 }, { 4, 8 } }), assume_unique: true);
            x3.data.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L, 5L, 6L);

            var u = np.ma.union1d(L(new long[] { 1, 2, 5, 7, 5, -1 }, new[] { 0, 0, 0, 0, 0, 1 }),
                                  L(new long[] { 1, 2, 3, 4, 5, -1 }, new[] { 0, 0, 0, 0, 0, 1 }));
            u.data.ToArray<long>()[..6].Should().Equal(1L, 2L, 3L, 4L, 5L, 7L);
            Mask(u).Should().Equal(false, false, false, false, false, false, true);
            np.ma.union1d(np.array(new long[,] { { 1, 2 }, { 3, 4 } }), np.array(new long[] { 4, 5 })).data.ToArray<long>()
              .Should().Equal(1L, 2L, 3L, 4L, 5L);

            var s = np.ma.setdiff1d(L(new long[] { 1, 2, 3, 4, 5, 6, 7 }, new[] { 0, 0, 0, 1, 0, 0, 1 }),
                                    L(new long[] { 2, 4, 3, 3, 2, 1, 5 }, new[] { 0, 0, 0, 0, 1, 0, 0 }));
            s.data.ToArray<long>().Should().Equal(6L);
            Mask(s).Should().Equal(false);
        }

        /// <summary>
        ///     setxor1d compares the joined uniques FILLED with the default fill value, so a masked slot equals an unmasked
        ///     999999 (the int64 default) and both drop out: <c>setxor1d([999999, 2], ma.array([5, 7], mask=[1, 0]))</c> is
        ///     [2, 7].
        /// </summary>
        [TestMethod]
        public void Setxor1d_ComparesTheFilledValues()
        {
            var r = np.ma.setxor1d(L(new long[] { 999999, 2 }), L(new long[] { 5, 7 }, new[] { 1, 0 }));
            r.data.ToArray<long>().Should().Equal(2L, 7L);
            Mask(r).Should().Equal(false, false);
        }

        /// <summary>
        ///     A 0-d operand: in1d's <c>flag[indx][rev_idx]</c> with a 0-d inverse selects a scalar — an unmasked 0-d result
        ///     (NumPy's np.bool_), or the masked constant; isin then reshapes that constant into a NEW 0-d float64 masked
        ///     array. With assume_unique the 0-d operand meets np.concatenate first.
        /// </summary>
        [TestMethod]
        public void In1d_ZeroD_ScalarOrMasked()
        {
            var r = np.ma.in1d(np.ma.array(NDArray.Scalar(5L)), L(new long[] { 5 }));
            r.ndim.Should().Be(0);
            ReferenceEquals(np.ma.getmask(r), np.ma.nomask).Should().BeTrue();
            r.data.GetBoolean().Should().BeTrue();

            var m = np.ma.in1d(np.ma.array(NDArray.Scalar(5L)), L(new long[] { 9 }, new[] { 1 }));
            ReferenceEquals(m, np.ma.masked).Should().BeTrue();

            var i = np.ma.isin(np.ma.array(NDArray.Scalar(5L)), L(new long[] { 9 }, new[] { 1 }));
            ReferenceEquals(i, np.ma.masked).Should().BeFalse();
            i.dtype.Should().Be(np.float64);
            i.ndim.Should().Be(0);
            Mask(i).Should().Equal(true);

            Action au = () => np.ma.in1d(np.ma.array(NDArray.Scalar(5L)), L(new long[] { 5 }), assume_unique: true);
            au.Should().Throw<ValueError>().WithMessage("zero-dimensional arrays cannot be concatenated");

            var e = np.ma.in1d(L(new long[0]), L(new long[0]));
            e.shape.Should().Equal(0L);
            e.dtype.Should().Be(np.@bool);
            ReferenceEquals(np.ma.getmask(e), np.ma.nomask).Should().BeTrue();
        }

        /// <summary><see cref="MaskedArrayModule.UniqueResult"/> stands in for NumPy's tuple: it deconstructs into the
        ///     present outputs, indexes them in NumPy's order and converts to the values.</summary>
        [TestMethod]
        public void UniqueResult_IsNumPysTuple()
        {
            var a = L(new long[] { 2, 1, 2 });
            var (u, inv) = np.ma.unique(a, return_inverse: true);
            u.data.ToArray<long>().Should().Equal(1L, 2L);
            inv.ToArray<long>().Should().Equal(1L, 0L, 1L);
            var (u2, idx) = np.ma.unique(a, return_index: true);
            idx.ToArray<long>().Should().Equal(1L, 0L);
            var both = np.ma.unique(a, true, true);
            both.Length.Should().Be(3);
            ((NDArray)both[1]).ToArray<long>().Should().Equal(1L, 0L);
            ((NDArray)both[2]).ToArray<long>().Should().Equal(1L, 0L, 1L);
            NDMaskedArray values = both;
            values.data.ToArray<long>().Should().Equal(1L, 2L);
            Action arity = () => { var (x, y) = both; };
            arity.Should().Throw<InvalidOperationException>().WithMessage("too many values to unpack (expected 2)");
            Action index = () => _ = np.ma.unique(a, return_inverse: true)[2];
            index.Should().Throw<IndexOutOfRangeException>();
        }
    }
}
