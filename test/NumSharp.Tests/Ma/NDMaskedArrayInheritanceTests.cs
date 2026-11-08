using System;
using System.Linq;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     numpy.ma's attribute and mask semantics for every result kind: what a result takes from its operand (the fill
    ///     value RAW and the hard-mask flag, through <c>_update_from</c> / <c>__array_finalize__</c>), whether its mask is
    ///     the operand's own object, a view of it or a new array, and the operator semantics MaskedArray does not define
    ///     itself (<c>-x</c>, <c>~x</c>, <c>%</c>, <c>&amp;</c>, <c>|</c>, <c>^</c> run ndarray's ufunc and
    ///     <c>__array_wrap__</c>; the ordering operators run <c>_comparison</c>). Every expected value was probed against
    ///     NumPy 2.4.2; a write test masks a slot through the result and reads the operand's mask back.
    /// </summary>
    [TestClass]
    public class NDMaskedArrayInheritanceTests
    {
        private static NDArray A(params double[] v) => np.array(v);
        private static NDArray L(params long[] v) => np.array(v);
        private static bool[] M(params int[] v) => v.Select(x => x != 0).ToArray();

        /// <summary>A hard-masked float64 array with a custom fill value, its first element masked.</summary>
        private static NDMaskedArray X() => np.ma.array(A(5, 7, 9, 4), mask: M(1, 0, 0, 0), fill_value: 7.5, hard_mask: true);

        private static double[] D(NDMaskedArray r) => np.ma.getdata(r).astype(np.float64).ToArray<double>();
        private static bool[] Mask(NDMaskedArray r) => np.ma.getmaskarray(r).ToArray<bool>();
        private static bool IsNomask(NDMaskedArray r) => ReferenceEquals(np.ma.getmask(r), np.ma.nomask);

        /// <summary>Masks element 1 of the result and returns the operand's mask afterwards.</summary>
        private static bool[] WriteThrough(NDMaskedArray operand, Func<NDMaskedArray, NDMaskedArray> op)
        {
            var result = op(operand);
            result[1] = np.ma.NDMasked;
            return Mask(operand);
        }

        // ---------------------------------------------------------------- the masked ufunc wrappers

        [TestMethod]
        public void UfuncResults_TakeTheFirstMaskedOperandsAttributes()
        {
            // ma.negative / ma.add(x, 1.) / ma.add(1., x) / ma.divide: fill 7.5, hard.
            foreach (var r in new[] { np.ma.negative(X()), np.ma.add(X(), 1.0), np.ma.add(1.0, X()), np.ma.divide(X(), 2.0), X() + 1.0, 1.0 - X() })
            {
                Convert.ToDouble(r.fill_value).Should().Be(7.5);
                r.hardmask.Should().BeTrue();
            }
            // The FIRST masked operand wins: soft (fill 3) + hard -> soft, fill 3.
            var soft = np.ma.array(np.ones(new Shape(4)), fill_value: 3.0);
            var r2 = np.ma.add(soft, X());
            Convert.ToDouble(r2.fill_value).Should().Be(3.0);
            r2.hardmask.Should().BeFalse();
            // The fill value is RAW: an int64 operand's 7 stays an int64 on sqrt's float64 result (NumPy's getter says
            // np.int64(7)), and a comparison FUNCTION keeps the float 7.5.
            var xi = np.ma.array(L(1, 2), mask: M(1, 0), fill_value: 7L, hard_mask: true);
            np.ma.sqrt(xi).fill_value.Should().Be(7L);
            Convert.ToDouble(np.ma.greater(X(), 2.0).fill_value).Should().Be(7.5);
        }

        [TestMethod]
        public void BinaryUfunc_BuildsANewMask()
        {
            // ma.add(x, 1.): `logical_or(ma, getmaskarray(b))` is a new array — masking the result leaves x alone.
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => np.ma.add(x, 1.0)).Should().Equal(true, false, false);
            // A domain-free unary shares the operand's mask object, as NumPy's does.
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => np.ma.negative(x)).Should().Equal(true, true, false);
        }

        [TestMethod]
        public void Power_InheritsFromTheBaseOnly_AndFillsInvalidSlotsWithItsFill()
        {
            // ma.power(plain, masked): soft, the default fill (result._update_from(a) of a plain base).
            var plainBase = np.ma.power(A(4, 9), np.ma.array(A(0.5, 0.5), mask: M(1, 0), fill_value: 7.0, hard_mask: true));
            plainBase.hardmask.Should().BeFalse();
            Convert.ToDouble(plainBase.fill_value).Should().Be(1e20);
            D(plainBase).Should().Equal(4.0, 3.0);
            // An invalid slot is masked and gets the RESULT's fill value — the base's: power([-1, 4], .5) -> [7, 2].
            var inv = np.ma.power(np.ma.array(A(-1, 4), fill_value: 7.0), 0.5);
            D(inv).Should().Equal(7.0, 2.0);
            Mask(inv).Should().Equal(true, false);
            // An all-False mask shrinks to nomask (mask_or) when nothing is invalid.
            IsNomask(np.ma.power(np.ma.array(A(4, 9), mask: M(0, 0), fill_value: 7.0), 0.5)).Should().BeTrue();
            // A 0-d power with a mask is masked.
            ReferenceEquals(np.ma.power(np.ma.array(4.0, mask: true), 0.5), np.ma.NDMasked).Should().BeTrue();
        }

        [TestMethod]
        public void Shift_ShiftsMaskedSlotsFromZero_AndInheritsNothing()
        {
            // ma.left_shift(ma.array([1, 2], mask=[1, 0], fill_value=7, hard_mask=True), 1) -> [0, 4], soft, fill 999999.
            var r = np.ma.left_shift(np.ma.array(L(1, 2), mask: M(1, 0), fill_value: 7L, hard_mask: true), 1);
            np.ma.getdata(r).ToArray<long>().Should().Equal(0L, 4L);
            r.hardmask.Should().BeFalse();
            Convert.ToInt64(r.fill_value).Should().Be(999999L);
            // `masked_array(d, mask=m)` adopts the operand's mask object.
            WriteThrough(np.ma.array(L(1, 2, 3), mask: M(1, 0, 0)), x => np.ma.left_shift(x, 1)).Should().Equal(true, true, false);
            // A masked shift count: its mask and its fill value.
            var c = np.ma.left_shift(np.ma.array(L(1, 2)), np.ma.array(L(1, 2), mask: M(0, 1), fill_value: 5L));
            np.ma.getdata(c).ToArray<long>().Should().Equal(2L, 8L);
            Mask(c).Should().Equal(false, true);
            Convert.ToInt64(c.fill_value).Should().Be(5L);
        }

        // ---------------------------------------------------------------- operators

        [TestMethod]
        public void UnaryOperators_AreTheUfuncWithoutFillBack()
        {
            // -x: the masked slot holds -5 (ma.negative would write 5 back); the mask is x's own object; fill and hardness kept.
            var neg = -X();
            D(neg).Should().Equal(-5.0, -7.0, -9.0, -4.0);
            Convert.ToDouble(neg.fill_value).Should().Be(7.5);
            neg.hardmask.Should().BeTrue();
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => -x).Should().Equal(true, true, false);
            // A nomask operand gives an all-False mask ARRAY; a masked 0-d operand gives the masked constant.
            var plain = -np.ma.array(A(1, 2));
            IsNomask(plain).Should().BeFalse();
            Mask(plain).Should().Equal(false, false);
            ReferenceEquals(-np.ma.array(1.0, mask: true), np.ma.NDMasked).Should().BeTrue();
            // ~x of int64 [1, 2] masked at 0: [-2, -3].
            np.ma.getdata(~np.ma.array(L(1, 2), mask: M(1, 0), fill_value: 7L)).ToArray<long>().Should().Equal(-2L, -3L);
        }

        [TestMethod]
        public void ModuloAndBitwiseOperators_AreTheUfuncWithArrayWrap()
        {
            var y = np.ma.array(A(2, 0, 4, 3), mask: M(0, 0, 1, 0), fill_value: 3.5);
            // X % Y: remainder of the data, 1 where |X| * tiny >= |Y| (7 % 0), mask [T, T, T, F]; ma.mod keeps [5, 7, 9, 1].
            var mod = X() % y;
            D(mod).Should().Equal(1.0, 1.0, 1.0, 1.0);
            Mask(mod).Should().Equal(true, true, true, false);
            Convert.ToDouble(mod.fill_value).Should().Be(7.5);
            mod.hardmask.Should().BeTrue();
            D(np.ma.mod(X(), y)).Should().Equal(5.0, 7.0, 9.0, 1.0);
            // A masked 0-d result is masked; an all-False mask_or shrinks to nomask.
            ReferenceEquals(np.ma.array(5.0) % 0.0, np.ma.NDMasked).Should().BeTrue();
            IsNomask(np.ma.array(L(5, 6), mask: M(0, 0)) & 3).Should().BeTrue();
            // x & 3: the masked slot holds 5 & 3 = 1 (no fill-back), fill 9 and hardness kept.
            var and = np.ma.array(L(5, 6, 7), mask: M(1, 0, 0), fill_value: 9L, hard_mask: true) & 3;
            np.ma.getdata(and).ToArray<long>().Should().Equal(1L, 2L, 3L);
            Convert.ToInt64(and.fill_value).Should().Be(9L);
            and.hardmask.Should().BeTrue();
            // The operands stay weak: float32 % 2.0 is float32.
            (np.ma.array(np.array(new float[] { 5.5f, 7.5f })) % 2.0).dtype.Should().Be(np.float32);
            // The mask is a new array.
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => x % 3.0).Should().Equal(true, false, false);
        }

        [TestMethod]
        public void OrderingOperators_AreNumPysComparison()
        {
            // X > 6.0: the masked slot compares its own data (5 > 6 is False) — no fill-back —; the fill 7.5 casts to True.
            var gt = X() > 6.0;
            np.ma.getdata(gt).ToArray<bool>().Should().Equal(false, true, true, false);
            Mask(gt).Should().Equal(true, false, false, false);
            gt.fill_value.Should().Be(true);
            gt.hardmask.Should().BeTrue();
            // The ufunc function writes the operand's data back: ma.greater(X, 6.0) holds bool(5) = True at the masked slot.
            np.ma.getdata(np.ma.greater(np.ma.array(A(1, 7), mask: M(1, 0)), 6.0)).ToArray<bool>().Should().Equal(true, true);
            np.ma.getdata(np.ma.array(A(1, 7), mask: M(1, 0)) > 6.0).ToArray<bool>().Should().Equal(false, true);
            // An all-False mask shrinks to nomask; a fill of 0.0 is False, NaN is True.
            IsNomask(np.ma.array(A(1, 2), mask: M(0, 0)) > 1.5).Should().BeTrue();
            (np.ma.array(A(1, 2), mask: M(1, 0), fill_value: 0.0) > 1.0).fill_value.Should().Be(false);
            (np.ma.array(A(1, 2), mask: M(1, 0), fill_value: double.NaN) > 1.0).fill_value.Should().Be(true);
            // getdata(0.1) is a STRONG float64: float32 [0.1, 0.5] > 0.1 compares in float64 -> [True, True].
            np.ma.getdata(np.ma.array(np.array(new float[] { 0.1f, 0.5f })) > 0.1).ToArray<bool>().Should().Equal(true, true);
            // 0-d: masked -> the constant; unmasked -> the value.
            ReferenceEquals(np.ma.array(1.0, mask: true) > 0, np.ma.NDMasked).Should().BeTrue();
            np.ma.getdata(np.ma.array(1.0) > 0).GetBoolean().Should().BeTrue();
            // Reflected: 1.5 <= X is X.__ge__(1.5) -> [1, 1, 1, 1], X's hardness.
            var le = 1.5 <= X();
            np.ma.getdata(le).ToArray<bool>().Should().Equal(true, true, true, true);
            le.hardmask.Should().BeTrue();
            // The mask is a copy.
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => x > 1.0).Should().Equal(true, false, false);
        }

        // ---------------------------------------------------------------- methods

        [TestMethod]
        public void CopyAstypeView_FollowArrayFinalize()
        {
            var c = np.ma.copy(X());
            c.hardmask.Should().BeTrue();
            Convert.ToDouble(c.fill_value).Should().Be(7.5);
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => np.ma.copy(x)).Should().Equal(true, false, false);
            // astype: hard, the mask COPIED; copy=False of the same dtype is the array itself.
            X().astype(np.float32).hardmask.Should().BeTrue();
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => x.astype(np.float32)).Should().Equal(true, false, false);
            var x0 = X();
            ReferenceEquals(x0.astype(np.float64, copy: false), x0).Should().BeTrue();
            // view: hard, the mask a VIEW; a dtype resets a stored fill (int64's 999999), even the same dtype.
            X().view().hardmask.Should().BeTrue();
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => x.view()).Should().Equal(true, true, false);
            Convert.ToInt64(np.ma.array(A(1, 2), fill_value: 7.0).view(np.int64).fill_value).Should().Be(999999L);
            Convert.ToDouble(np.ma.array(A(1, 2), fill_value: 7.0).view(np.float64).fill_value).Should().Be(1e20);
            // fill_value= replaces a STORED fill and is ignored otherwise.
            Convert.ToDouble(np.ma.array(A(1, 2), mask: M(1, 0), fill_value: 7.0).view(fill_value: 5.0).fill_value).Should().Be(5.0);
            Convert.ToDouble(np.ma.array(A(1, 2), mask: M(1, 0)).view(fill_value: 5.0).fill_value).Should().Be(1e20);
        }

        [TestMethod]
        public void DiagonalClipRoundRealMT_InheritOrResetLikeNumPy()
        {
            var two = np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), mask: new[,] { { true, false }, { false, false } }, fill_value: 7.0, hard_mask: true);
            var diag = np.ma.diagonal(two);
            diag.hardmask.Should().BeTrue();
            Convert.ToDouble(diag.fill_value).Should().Be(7.0);
            // clip inherits and COPIES the mask; round inherits and SHARES it.
            np.ma.clip(X(), 1.5, 3.5).hardmask.Should().BeTrue();
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => np.ma.clip(x, 0, 1)).Should().Equal(true, false, false);
            Convert.ToDouble(np.ma.round(X(), 1).fill_value).Should().Be(7.5);
            WriteThrough(np.ma.array(A(1.5, 2.5, 3.5), mask: M(1, 0, 0)), x => np.ma.round(x)).Should().Equal(true, true, false);
            // real / imag: soft, the default fill, a copied mask; mT: soft, the default fill, a mask VIEW.
            var re = X().real;
            re.hardmask.Should().BeFalse();
            Convert.ToDouble(re.fill_value).Should().Be(1e20);
            WriteThrough(np.ma.array(A(1, 2, 3), mask: M(1, 0, 0)), x => x.real).Should().Equal(true, false, false);
            var mt = two.mT;
            mt.hardmask.Should().BeFalse();
            Convert.ToDouble(mt.fill_value).Should().Be(1e20);
            var src = np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), mask: new[,] { { true, false }, { false, false } });
            var t = src.mT;
            t[0, 1] = np.ma.NDMasked;
            Mask(src).Should().Equal(true, false, true, false);
            // anom inherits (self - mean).
            np.ma.anom(np.ma.array(A(1, 2), fill_value: 7.0, hard_mask: true)).hardmask.Should().BeTrue();
        }

        [TestMethod]
        public void Sort_WritesBackThroughTheHardMask()
        {
            // ma.sort of a HARD-masked [3, 1(m), 2]: `self[...] = take_along_axis(...)` keeps the masked slot's data and ORs
            // the masks -> [2, 1, 2], mask [F, T, T]. Soft: [2, 3, 1], mask [F, F, T].
            var hard = np.ma.sort(np.ma.array(A(3, 1, 2), mask: M(0, 1, 0), fill_value: 7.0, hard_mask: true));
            D(hard).Should().Equal(2.0, 1.0, 2.0);
            Mask(hard).Should().Equal(false, true, true);
            hard.hardmask.Should().BeTrue();
            var soft = np.ma.sort(np.ma.array(A(3, 1, 2), mask: M(0, 1, 0), fill_value: 7.0));
            D(soft).Should().Equal(2.0, 3.0, 1.0);
            Mask(soft).Should().Equal(false, false, true);
            Convert.ToDouble(soft.fill_value).Should().Be(7.0);
            // axis=None flattens: [[1(m), 2], [3, 4]] hard -> [1, 3, 4, 4], mask [T, F, F, T].
            var flat = np.ma.sort(np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), mask: new[,] { { true, false }, { false, false } }, hard_mask: true), null);
            D(flat).Should().Equal(1.0, 3.0, 4.0, 4.0);
            Mask(flat).Should().Equal(true, false, false, true);
            // A float masked slot sorts by NaN ("nan > inf"): it lands after +inf data.
            np.ma.argsort(np.ma.array(A(double.PositiveInfinity, 1, double.PositiveInfinity, 2), mask: M(1, 0, 0, 0))).ToArray<long>()
              .Should().Equal(1L, 3L, 2L, 0L);
        }

        [TestMethod]
        public void DiffDotMaskedWhere_FollowNumPysStatements()
        {
            // diff inherits; a 0-d prepend broadcasts (and numpy.ma's concatenate gives the defaults); n=0 is the operand.
            var d1 = np.ma.diff(np.ma.array(A(1, 3, 6), fill_value: 7.0, hard_mask: true));
            D(d1).Should().Equal(2.0, 3.0);
            d1.hardmask.Should().BeTrue();
            var pre = np.ma.diff(np.ma.array(A(1, 3), mask: M(0, 1), fill_value: 7.0, hard_mask: true), prepend: 0.0);
            D(pre).Should().Equal(1.0, 2.0);
            Mask(pre).Should().Equal(false, true);
            pre.hardmask.Should().BeFalse();
            var x = np.ma.array(A(1, 3));
            ReferenceEquals(np.ma.diff(x, 0), x).Should().BeTrue();
            // dot: an empty contraction masks every slot; an array result always carries a mask array; a masked 0-d result
            // is a 0-d array with a True mask, not the constant.
            var empty = np.ma.dot(np.ma.array(np.zeros(new Shape(2, 0))), np.ma.array(np.zeros(new Shape(0, 3))));
            Mask(empty).Should().OnlyContain(b => b);
            IsNomask(np.ma.dot(np.ma.array(np.array(new double[,] { { 1, 2 } })), np.ma.array(np.array(new double[,] { { 1 }, { 2 } })))).Should().BeFalse();
            var zero = np.ma.dot(np.ma.array(A(1, 2), mask: M(1, 1)), np.ma.array(A(1, 2)));
            ReferenceEquals(zero, np.ma.NDMasked).Should().BeFalse();
            np.ma.getmaskarray(zero).GetBoolean().Should().BeTrue();
            // masked_where(copy=False) masks the INPUT in place.
            var input = np.ma.array(A(1, 2, 3), mask: M(1, 0, 0));
            var masked = np.ma.masked_where(new[] { false, true, false }, input, copy: false);
            Mask(input).Should().Equal(true, true, false);
            masked[2] = np.ma.NDMasked;
            Mask(input).Should().Equal(true, true, true);
        }
    }
}
