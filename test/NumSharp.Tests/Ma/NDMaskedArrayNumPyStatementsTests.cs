using System;
using System.Linq;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     numpy.ma's own STATEMENTS for the members the string-oracle validation (the <c>strma</c> section of
    ///     <c>strings.jsonl</c>) found NumSharp answering differently for every dtype, not only for text: what a result
    ///     inherits (fill value, hard mask), when a mask is an array and when <see cref="MaskedArrayModule.nomask"/>, which
    ///     results are VIEWS, what the logical reductions and <c>count</c> return without a mask, and the statement order
    ///     of <c>trace</c>, <c>allclose</c>, <c>average</c>, the two-operand extrema and <c>unique</c>. Every expected value
    ///     was probed against NumPy 2.4.2; each test pins one rule.
    /// </summary>
    [TestClass]
    public class NDMaskedArrayNumPyStatementsTests
    {
        /// <summary>A hard-masked int64 array with a custom fill value: the source every inheritance test reads.</summary>
        private static NDMaskedArray Inheriting() =>
            np.ma.array(np.array(new long[] { 1, 2, 3 }), mask: new[] { true, false, false }, fill_value: 7L, hard_mask: true);

        private static long[] Data(NDMaskedArray r) => np.ma.getdata(r).ToArray<long>();
        private static bool[] Mask(NDMaskedArray r) => np.ma.getmaskarray(r).ToArray<bool>();
        private static bool IsNomask(NDMaskedArray r) => ReferenceEquals(np.ma.getmask(r), np.ma.nomask);

        // ---------------------------------------------------------------- what a result inherits

        [TestMethod]
        public void ShapeViews_InheritTheFillValueAndTheHardMask()
        {
            // ma.ravel(x) / ma.reshape(x, (3, 1)) / ma.transpose(x): mask kept, fill 7, hard (the _arraymethod path).
            foreach (var r in new[] { np.ma.ravel(Inheriting()), np.ma.reshape(Inheriting(), 3, 1), np.ma.transpose(Inheriting()) })
            {
                Mask(r).Should().Equal(true, false, false);
                Convert.ToInt64(r.fill_value).Should().Be(7L);
                r.hardmask.Should().BeTrue();
            }
        }

        [TestMethod]
        public void Take_DoesNotInherit()
        {
            // ma.take(x, [0, 1]) -> data [1, 2], mask [True, False], fill 999999, soft.
            var r = np.ma.take(Inheriting(), np.array(new long[] { 0, 1 }));
            Data(r).Should().Equal(1L, 2L);
            Mask(r).Should().Equal(true, false);
            Convert.ToInt64(r.fill_value).Should().Be(999999L);
            r.hardmask.Should().BeFalse();
        }

        [TestMethod]
        public void LikeFunctions_CopyTheMask_AndInheritFillAndHardness()
        {
            // ma.zeros_like(x) -> [0, 0, 0], mask [T, F, F], fill 7, hard; ma.ones_like(x, float64) -> fill 7.0.
            var z = np.ma.zeros_like(Inheriting());
            Data(z).Should().Equal(0L, 0L, 0L);
            Mask(z).Should().Equal(true, false, false);
            Convert.ToInt64(z.fill_value).Should().Be(7L);
            z.hardmask.Should().BeTrue();

            var o = np.ma.ones_like(Inheriting(), np.float64);
            np.ma.getdata(o).ToArray<double>().Should().Equal(1.0, 1.0, 1.0);
            Mask(o).Should().Equal(true, false, false);
            Convert.ToDouble(o.fill_value).Should().Be(7.0);
            o.hardmask.Should().BeTrue();

            // The mask is a COPY: masking the result leaves the source alone.
            var source = Inheriting();
            var like = np.ma.zeros_like(source);
            like[1] = np.ma.NDMasked;
            Mask(source).Should().Equal(true, false, false);

            // A plain operand has no mask to copy: ma.empty_like(np.zeros(2)).mask is nomask.
            IsNomask(np.ma.empty_like(np.zeros(2))).Should().BeTrue();

            // ma.masked_all_like(x): every element masked, fill 7, hard (the data is uninitialized).
            var all = np.ma.masked_all_like(Inheriting());
            Mask(all).Should().Equal(true, true, true);
            Convert.ToInt64(all.fill_value).Should().Be(7L);
            all.hardmask.Should().BeTrue();
        }

        [TestMethod]
        public void Asarray_Inherits_Asanyarray_ReturnsTheOperandItself()
        {
            var a = np.ma.asarray(Inheriting());
            Convert.ToInt64(a.fill_value).Should().Be(7L);
            a.hardmask.Should().BeTrue();

            // ma.asanyarray(x) is x; a dtype change is a new array.
            var x = Inheriting();
            ReferenceEquals(np.ma.asanyarray(x), x).Should().BeTrue();
            ReferenceEquals(np.ma.asanyarray(x, np.float64), x).Should().BeFalse();
        }

        [TestMethod]
        public void Array_IsSoftByDefault_MaskedArrayInheritsTheHardness()
        {
            // ma.array (the function, hard_mask=False) vs ma.masked_array (the class, hard_mask=None inherits).
            var soft = np.ma.array(Inheriting());
            soft.hardmask.Should().BeFalse();
            Convert.ToInt64(soft.fill_value).Should().Be(7L);
            Mask(soft).Should().Equal(true, false, false);
            np.ma.masked_array(Inheriting()).hardmask.Should().BeTrue();
        }

        // ---------------------------------------------------------------- masks: adopted, viewed, array or nomask

        [TestMethod]
        public void Array_AdoptsTheCallersBooleanMask()
        {
            // m = np.array([True, False, False]); x = ma.array(np.arange(3), mask=m); x[1] = ma.masked
            // -> m is [True, True, False] and ma.getmask(x) is m.
            var m = np.array(new[] { true, false, false });
            var x = np.ma.array(np.arange(3), mask: m);
            x[1] = np.ma.NDMasked;
            m.ToArray<bool>().Should().Equal(true, true, false);
            ReferenceEquals(np.ma.getmask(x), m).Should().BeTrue();

            // copy=True copies it; a non-bool mask is converted (a new array).
            var kept = np.array(new[] { true, false, false });
            var copied = np.ma.array(np.arange(3), mask: kept, copy: true);
            copied[1] = np.ma.NDMasked;
            kept.ToArray<bool>().Should().Equal(true, false, false);
            var ints = np.array(new[] { 1, 0, 0 });
            ReferenceEquals(np.ma.getmask(np.ma.array(np.arange(3), mask: ints)), ints).Should().BeFalse();
        }

        [TestMethod]
        public void Array_OfAMaskedArray_SharesItsMaskThroughAView()
        {
            // y = ma.array(src): getmask(y) is not getmask(src), shares memory with it, OWNDATA False.
            var src = np.ma.array(np.arange(3), mask: new[] { true, false, false });
            var y = np.ma.array(src);
            ReferenceEquals(np.ma.getmask(y), np.ma.getmask(src)).Should().BeFalse();
            np.shares_memory(np.ma.getmask(y), np.ma.getmask(src)).Should().BeTrue();
            np.ma.getmask(y).flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void FromNxFunctions_AlwaysBuildAMaskArray_AndInheritNothing()
        {
            // ma.atleast_1d / atleast_2d / diagflat / hstack / stack over unmasked operands: a False mask ARRAY.
            var a1 = np.ma.atleast_1d(np.ma.array(np.array(new long[] { 1, 2 })));
            IsNomask(a1).Should().BeFalse();
            Mask(a1).Should().Equal(false, false);

            var a2 = np.ma.atleast_2d(np.ma.array(5L));
            a2.shape.Should().Equal(new long[] { 1, 1 });
            IsNomask(a2).Should().BeFalse();

            var d = np.ma.diagflat(np.ma.array(np.array(new long[] { 1, 2 })));
            Data(d).Should().Equal(1L, 0L, 0L, 2L);
            Mask(d).Should().Equal(false, false, false, false);
            IsNomask(d).Should().BeFalse();

            var h = np.ma.hstack(np.ma.array(np.array(new long[] { 1 })), np.ma.array(np.array(new long[] { 2 })));
            IsNomask(h).Should().BeFalse();

            // ma.stack([ma.array(5), ma.array(6)]) -> shape (2,), mask [False, False].
            var s = np.ma.stack(new object[] { np.ma.array(5L), np.ma.array(6L) });
            s.shape.Should().Equal(new long[] { 2 });
            Data(s).Should().Equal(5L, 6L);
            IsNomask(s).Should().BeFalse();

            // ma.vstack([x, x]) keeps the masks but not the fill value or the hardness.
            var v = np.ma.vstack(Inheriting(), Inheriting());
            Mask(v).Should().Equal(true, false, false, true, false, false);
            Convert.ToInt64(v.fill_value).Should().Be(999999L);
            v.hardmask.Should().BeFalse();
        }

        [TestMethod]
        public void Concatenate_ShrinksAnAllFalseMask()
        {
            // ma.concatenate([ma.array([1, 2], mask=[0, 0]), ma.array([3])]) -> nomask; with a masked element the mask
            // is kept; two empty masked operands -> nomask.
            var shrunk = np.ma.concatenate(new object[]
            {
                np.ma.array(np.array(new long[] { 1, 2 }), mask: new[] { false, false }),
                np.ma.array(np.array(new long[] { 3 })),
            });
            Data(shrunk).Should().Equal(1L, 2L, 3L);
            IsNomask(shrunk).Should().BeTrue();

            var kept = np.ma.concatenate(new object[]
            {
                np.ma.array(np.array(new long[] { 1, 2 }), mask: new[] { false, true }),
                np.ma.array(np.array(new long[] { 3 })),
            });
            Mask(kept).Should().Equal(false, true, false);

            var empty = np.ma.array(np.zeros(new Shape(0L)), mask: np.zeros(new Shape(0L), np.@bool));
            IsNomask(np.ma.concatenate(new object[] { empty, empty })).Should().BeTrue();
        }

        [TestMethod]
        public void ShrinkMask_IsInPlace_AndReturnsTheArrayItself()
        {
            // xs.shrink_mask() is xs, and xs._mask is nomask afterwards.
            var xs = np.ma.array(np.array(new long[] { 1, 2 }), mask: new[] { false, false });
            ReferenceEquals(xs.shrink_mask(), xs).Should().BeTrue();
            IsNomask(xs).Should().BeTrue();
        }

        [TestMethod]
        public void HardenMask_OfAPlainArray_IsAHardUnmaskedArray()
        {
            var h = np.ma.harden_mask(np.array(new long[] { 1, 2 }));
            h.hardmask.Should().BeTrue();
            IsNomask(h).Should().BeTrue();
        }

        // ---------------------------------------------------------------- results that are views

        [TestMethod]
        public void Getdata_Filled_Compressed_CompressNd_ReturnViewsOfUnmaskedData()
        {
            var d = np.arange(3);
            // ma.getdata(ma.array(d)) shares memory with d (a view, OWNDATA False).
            var data = np.ma.getdata(np.ma.array(d));
            np.shares_memory(data, d).Should().BeTrue();
            data.flags.owndata.Should().BeFalse();
            // filled() with nomask, or with nothing masked, is the data itself (a view).
            np.shares_memory(np.ma.array(d).filled(), d).Should().BeTrue();
            np.shares_memory(np.ma.array(d, mask: new[] { false, false, false }).filled(), d).Should().BeTrue();
            // compressed() without a mask is ndarray.ravel(self._data) — a view of contiguous data.
            np.shares_memory(np.ma.compressed(np.ma.array(d)), d).Should().BeTrue();
            // compress_nd with nothing masked returns x._data.
            var d2 = np.arange(3).reshape(1, 3);
            var nd = np.ma.compress_nd(np.ma.array(d2));
            nd.shape.Should().Equal(new long[] { 1, 3 });
            np.shares_memory(nd, d2).Should().BeTrue();
        }

        // ---------------------------------------------------------------- logical reductions and count

        [TestMethod]
        public void AlltrueAndSometrue_AreLogicalReductionsAlongAxisZero()
        {
            var x = np.array(new long[,] { { 1, 0 }, { 1, 1 } });
            // ma.alltrue(ma.array([[1, 0], [1, 1]])) -> [True, False], nomask.
            var plain = np.ma.alltrue(np.ma.array(x));
            np.ma.getdata(plain).ToArray<bool>().Should().Equal(true, false);
            IsNomask(plain).Should().BeTrue();
            // With a mask the masked slot is True and the mask stays an ARRAY: [True, True], mask [False, False].
            var masked = np.ma.alltrue(np.ma.array(x, mask: new[,] { { false, true }, { false, false } }));
            np.ma.getdata(masked).ToArray<bool>().Should().Equal(true, true);
            IsNomask(masked).Should().BeFalse();
            Mask(masked).Should().Equal(false, false);
            // A 1-D operand reduces to a scalar; a fully masked one to the masked constant; axis=None flattens.
            var one = np.ma.alltrue(np.ma.array(np.array(new long[] { 1, 0, 1 }), mask: new[] { false, true, false }));
            one.ndim.Should().Be(0);
            np.ma.getdata(one).GetBoolean().Should().BeTrue();
            ReferenceEquals(np.ma.alltrue(np.ma.array(np.array(new long[] { 1, 0, 1 }), mask: true)), np.ma.NDMasked).Should().BeTrue();
            np.ma.getdata(np.ma.alltrue(np.ma.array(x), null)).GetBoolean().Should().BeFalse();
            // ma.sometrue(ma.array([[1, 0], [0, 0]], mask=[[1, 0], [0, 0]])) -> [False, False], mask [False, False].
            var some = np.ma.sometrue(np.ma.array(np.array(new long[,] { { 1, 0 }, { 0, 0 } }), mask: new[,] { { true, false }, { false, false } }));
            np.ma.getdata(some).ToArray<bool>().Should().Equal(false, false);
            Mask(some).Should().Equal(false, false);
        }

        [TestMethod]
        public void Alltrue_HonoursTheDtypeOnlyWithAMask()
        {
            // Without a mask the dtype is not passed: [True, False]. With one, logical_and has no int64 loop.
            var x = np.array(new long[,] { { 1, 0 }, { 1, 1 } });
            np.ma.getdata(np.ma.alltrue(np.ma.array(x), 0, np.int64)).ToArray<bool>().Should().Equal(true, false);
            Action masked = () => np.ma.alltrue(np.ma.array(x, mask: new[,] { { false, true }, { false, false } }), 0, np.int64);
            masked.Should().Throw<TypeError>()
                  .WithMessage("No loop matching the specified signature and casting was found for ufunc logical_and");
        }

        [TestMethod]
        public void Count_WithoutAMask_IsAWriteableFullArray()
        {
            // ma.count(ma.array([1, 2, 3]), axis=0) -> 3 (0-d int64, OWNDATA, writeable);
            // ma.count(2-D, axis=1, keepdims=True) -> [[3], [3]].
            var c = np.ma.count(np.ma.array(np.array(new long[] { 1, 2, 3 })), 0);
            c.ndim.Should().Be(0);
            c.dtype.Should().Be(np.int64);
            c.GetInt64().Should().Be(3L);
            c.flags.writeable.Should().BeTrue();
            var k = np.ma.count(np.ma.array(np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } })), 1, keepdims: true);
            k.shape.Should().Equal(new long[] { 2, 1 });
            k.ToArray<long>().Should().Equal(3L, 3L);
            k.flags.writeable.Should().BeTrue();
            // A 0-d operand has no axis 1.
            Action bad = () => np.ma.count(np.ma.array(5L), 1);
            bad.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 0*");
        }

        [TestMethod]
        public void FlatnotmaskedEdges_FollowNumPysThreeCases()
        {
            // empty -> [0, -1]; nothing masked -> [0, size - 1]; some -> first/last unmasked; all masked -> None.
            np.ma.flatnotmasked_edges(np.ma.array(np.zeros(new Shape(0L)))).ToArray<long>().Should().Equal(0L, -1L);
            np.ma.flatnotmasked_edges(np.ma.array(np.array(new long[] { 1, 2, 3 }), mask: new[] { false, false, false }))
              .ToArray<long>().Should().Equal(0L, 2L);
            np.ma.flatnotmasked_edges(np.ma.array(np.array(new long[] { 1, 2, 3, 4 }), mask: new[] { true, false, false, true }))
              .ToArray<long>().Should().Equal(1L, 2L);
            np.ma.flatnotmasked_edges(np.ma.array(np.array(new long[] { 1, 2 }), mask: true)).Should().BeNull();
        }

        // ---------------------------------------------------------------- statement order

        [TestMethod]
        public void Trace_CastsTheDiagonalFirstWhenThereIsAMask()
        {
            // Diagonal [2**53, 1, 1]. nomask: trace() in int64 = 2**53 + 2, then astype(None) -> 9007199254740994.0.
            // A mask (even all False): D.astype(f8).filled(0).sum(-1) adds in float64 -> 9007199254740992.0.
            long big = 1L << 53;
            var m3 = np.array(new long[,] { { big, 5, 6 }, { 7, 1, 8 }, { 9, 10, 1 } });
            np.ma.trace(np.ma.array(m3)).GetDouble().Should().Be(9007199254740994.0);
            np.ma.trace(np.ma.array(m3, mask: np.zeros(new Shape(3L, 3L), np.@bool))).GetDouble().Should().Be(9007199254740992.0);
            // dtype=int64 casts nothing that loses precision on either path: 9007199254740994.
            np.ma.trace(np.ma.array(m3, mask: np.zeros(new Shape(3L, 3L), np.@bool)), dtype: np.int64).GetInt64().Should().Be(9007199254740994L);
            // A masked diagonal element counts as 0: [[1, 2], [3, 4]] with [1, 1] masked -> 1.0.
            np.ma.trace(np.ma.array(np.array(new long[,] { { 1, 2 }, { 3, 4 } }), mask: new[,] { { false, false }, { false, true } }))
              .GetDouble().Should().Be(1.0);
        }

        [TestMethod]
        public void Extrema_TakeTwoOperands_AndAreWhereOfAComparison()
        {
            var a = np.ma.array(np.array(new long[] { 1, 5, 3 }), mask: new[] { false, false, true });
            var b = np.array(new long[] { 4, 2, 9 });
            // ma.minimum(a, b) = where(less(a, b), a, b): a masked comparison takes b's data -> [1, 2, 9], mask [F, F, T].
            var min = np.ma.minimum(a, b);
            Data(min).Should().Equal(1L, 2L, 9L);
            Mask(min).Should().Equal(false, false, true);
            Data(np.ma.maximum(a, b)).Should().Equal(4L, 5L, 9L);
            // Swapped, the masked slot takes a's data: [1, 2, 3].
            Data(np.ma.minimum(b, a)).Should().Equal(1L, 2L, 3L);
            // less(NaN, 2.0) is False, so the NaN does NOT propagate (np.minimum would): [1.0, 2.0, 2.0].
            var nan = np.ma.minimum(np.ma.array(np.array(new[] { 1.0, double.NaN, 3.0 }), mask: new[] { false, false, false }),
                                    np.array(new[] { 2.0, 2.0, 2.0 }));
            np.ma.getdata(nan).ToArray<double>().Should().Equal(1.0, 2.0, 2.0);
        }

        [TestMethod]
        public void Allclose_FollowsNumPysStatements()
        {
            // A masked slot compares equal (masked_equal=True); NaN never equals NaN unmasked; infinities must agree in
            // place AND sign.
            np.ma.allclose(np.ma.array(np.array(new long[] { 1, 2, 3 }), mask: new[] { false, true, false }), np.array(new long[] { 1, 7, 3 }))
              .Should().BeTrue();
            np.ma.allclose(np.ma.array(np.array(new[] { double.NaN, 1.0 }), mask: new[] { true, false }), np.array(new[] { double.NaN, 1.0 }))
              .Should().BeTrue();
            np.ma.allclose(np.array(new[] { double.NaN, 1.0 }), np.array(new[] { double.NaN, 1.0 })).Should().BeFalse();
            np.ma.allclose(np.array(new[] { double.PositiveInfinity, 1.0 }), np.array(new[] { double.PositiveInfinity, 1.0 })).Should().BeTrue();
            np.ma.allclose(np.array(new[] { double.PositiveInfinity, 1.0 }), np.array(new[] { double.NegativeInfinity, 1.0 })).Should().BeFalse();
            np.ma.allclose(np.ma.array(np.array(new[] { 1.0, 9.0 }), mask: new[] { false, true }), np.array(new[] { 1.0, 2.0 }), masked_equal: false)
              .Should().BeFalse();
            // A 0-d x against a masked y: `masked_array(x, mask=m)` pairs the 0-d data with y's (3,) mask.
            Action mismatch = () => np.ma.allclose(5.0, np.ma.array(np.array(new[] { 1.0, 2.0, 3.0 }), mask: new[] { true, false, false }));
            mismatch.Should().Throw<MaskError>().WithMessage("Mask and data not compatible: data size is 1, mask size is 3.");
        }

        [TestMethod]
        public void Average_OfA0dMaskedOperand_WithWeights_IsNumPysAttributeError()
        {
            // `wgt * (~a.mask)` of two 0-d operands is a NumPy SCALAR, which has no `.mask` to update.
            Action f8 = () => np.ma.average(np.ma.array(5.0, mask: true), weights: np.ones(new Shape()));
            f8.Should().Throw<AttributeError>().WithMessage("'numpy.float64' object has no attribute 'mask'");
            Action i8 = () => np.ma.average(np.ma.array(5.0, mask: false), weights: np.ones(new Shape(), np.int64));
            i8.Should().Throw<AttributeError>().WithMessage("'numpy.int64' object has no attribute 'mask'");
        }

        [TestMethod]
        public void Setdiff1d_KeepsTheFirstOperandsDtype()
        {
            // ma.setdiff1d(int32 [3, 1, 2], float64 [2.0]) -> int32 [1, 3], nomask.
            var r = np.ma.setdiff1d(np.array(new[] { 3, 1, 2 }), np.array(new[] { 2.0 }));
            r.dtype.Should().Be(np.int32);
            np.ma.getdata(r).ToArray<int>().Should().Equal(1, 3);
            IsNomask(r).Should().BeTrue();
        }

        [TestMethod]
        public void Unique_KeepsTheFirstMaskedElementsData()
        {
            // ma.unique(ma.array([3, 1, 2, 1], mask=[0, 1, 0, 1])) -> data [2, 3, 1], mask [F, F, T].
            var r = np.ma.unique(np.ma.array(np.array(new long[] { 3, 1, 2, 1 }), mask: new[] { false, true, false, true }));
            Data(r).Should().Equal(2L, 3L, 1L);
            Mask(r).Should().Equal(false, false, true);
        }

        [TestMethod]
        public void Argsort_DefaultAxis_FlattensFromTwoDimensions()
        {
            // ma.argsort(2-D) with no axis sorts the FLATTENED array (NumPy's deprecated default): [1, 2, 0, 3].
            var two = np.ma.array(np.array(new long[,] { { 3, 1 }, { 2, 0 } }), mask: new[,] { { false, false }, { false, true } });
            var flat = np.ma.argsort(two);
            flat.shape.Should().Equal(new long[] { 4 });
            flat.ToArray<long>().Should().Equal(1L, 2L, 0L, 3L);
            // 1-D: the last axis -> [2, 0, 1]; an explicit axis 0 on 2-D -> [[1, 0], [0, 1]].
            np.ma.argsort(np.ma.array(np.array(new long[] { 3, 1, 2 }), mask: new[] { false, true, false })).ToArray<long>()
              .Should().Equal(2L, 0L, 1L);
            np.ma.argsort(two, 0).ToArray<long>().Should().Equal(1L, 0L, 0L, 1L);
            // The instance form has the same default.
            two.argsort().ToArray<long>().Should().Equal(1L, 2L, 0L, 3L);
        }
    }
}
