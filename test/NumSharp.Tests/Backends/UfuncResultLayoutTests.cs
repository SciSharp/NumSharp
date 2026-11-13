using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The layout of a fresh elementwise result (parity QA-36, fixed 2026-11-05): NumPy decides it in one place —
    ///     <c>try_trivial_single_output_loop</c>'s C / F allocation when the call qualifies (no mask, one output, no
    ///     input of two or more dimensions cast), NpyIter's K order otherwise (<c>npyiter_find_best_axis_ordering</c> +
    ///     <c>npyiter_new_temp_array</c>) — and every NumSharp engine now asks that rule
    ///     (<c>Shape.UfuncOutputShape</c>, <c>DefaultEngine.UfuncResultLayout.cs</c>). The broad matrix is the
    ///     <c>ufunc</c> family of the layout-parity corpus (<see cref="LayoutParityOracleTests"/>); these tests pin the
    ///     consequences a caller sees and the rule's corners, each expected value probed in NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class UfuncResultLayoutTests : TestClass
    {
        /// <summary>
        ///     The QA-36 reproduction: for <c>p = np.arange(24.).reshape(3, 2, 4).transpose(1, 0, 2)</c> (strides
        ///     <c>(32, 64, 8)</c>) <c>np.negative(p)</c> and <c>p + p</c> keep strides <c>(32, 64, 8)</c>, so
        ///     <c>np.ravel(np.negative(p), order='K')</c> reads <c>-0, -1, -2, …</c> (the memory order); over F operands the
        ///     float-tier binaries and cumprod along an axis keep F. NumSharp used to return C strides <c>(96, 32, 8)</c>, so
        ///     the K-order ravel read <c>-0, -1, -2, -3, -8, -9, …</c>.
        /// </summary>
        [TestMethod]
        public void UfuncResult_PermutedOperand_KeepsItsMemoryOrder()
        {
            using var p = np.arange(24.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });
            using var f = np.asfortranarray(np.arange(1.0, 25.0).reshape(2, 3, 4));

            using var neg = np.negative(p);
            using var sum = np.add(p, p);
            using var memoryOrder = np.ravel(neg, 'K');
            using var hyp = np.hypot(f, f);
            using var cp = np.cumprod(f, axis: 0);

            neg.strides.Should().Equal(32L, 64L, 8L);
            sum.strides.Should().Equal(32L, 64L, 8L);
            ((double)memoryOrder.GetValue(4)).Should().Be(-4.0, "a K-order ravel reads the result in memory order");
            hyp.flags.f_contiguous.Should().BeTrue("hypot of two F operands is F in NumPy");
            cp.flags.f_contiguous.Should().BeTrue("cumprod along an axis keeps an F operand's order");
        }

        /// <summary>
        ///     NpyIter's conflict rule: where two operands' memory orders disagree on a pair of axes, C order wins that pair
        ///     (<c>shouldswap</c> is cleared by any later operand voting C). An F operand beside a C one is C; a permuted
        ///     operand beside an F one keeps the permutation's order for the axes the F operand cannot overrule; an F operand
        ///     beside a (2, 3, 1) column votes F only on the axis the column abstains on. Probed 2.4.2 byte strides.
        /// </summary>
        [TestMethod]
        public void Conflicts_CWinsTheDisputedAxes()
        {
            using var c = np.arange(1.0, 25.0).reshape(2, 3, 4);
            using var f = np.asfortranarray(c);
            using var p = np.arange(1.0, 25.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });

            using var fc = f + c;
            using var pf = p + f;
            using var fcol = f + np.ones(new Shape(2, 3, 1));
            using var fmid = f + np.ones(new Shape(2, 1, 4));

            fc.strides.Should().Equal(96L, 32L, 8L);
            pf.strides.Should().Equal(32L, 64L, 8L);
            fcol.strides.Should().Equal(24L, 8L, 48L);
            fmid.strides.Should().Equal(96L, 8L, 24L);
        }

        /// <summary>
        ///     The two allocators differ only in the strides of extent-1 axes, and only <c>.strides</c> shows it: an int32 F
        ///     (2, 1, 3) array's <c>np.negative</c> takes the trivial loop (plain F strides (4, 8, 8)), its <c>np.sqrt</c> —
        ///     an int32 input cast to the float64 loop — and <c>x + 1.5</c> take NpyIter, which gives the extent-1 axis its
        ///     sorted place ((8, 48, 16)); <c>x + x</c> stays trivial. Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void ExtentOneAxes_TellTheTrivialLoopFromTheIterator()
        {
            using var x = np.asfortranarray(np.arange(6).astype(np.int32).reshape(2, 1, 3));

            using var neg = np.negative(x);
            using var root = np.sqrt(x);
            using var weak = x + 1.5;
            using var self = x + x;

            neg.strides.Should().Equal(4L, 8L, 8L);
            root.strides.Should().Equal(8L, 48L, 16L);
            weak.strides.Should().Equal(8L, 48L, 16L);
            self.strides.Should().Equal(4L, 8L, 8L);
        }

        /// <summary>
        ///     Negative strides: an allocated output never flips them, but NpyIter sorts by |stride|, so a flipped F operand
        ///     gives an F result (byte strides (8, 16, 48)) and a reversed C one a C result. The values are the operand's,
        ///     element for element.
        /// </summary>
        [TestMethod]
        public void FlippedOperands_SortByStrideMagnitude()
        {
            using var f = np.asfortranarray(np.arange(1.0, 25.0).reshape(2, 3, 4));
            using var flipped = np.flip(f, 2);
            using var reversed = np.arange(1.0, 25.0).reshape(2, 3, 4)[":, ::-1"];

            using var negFlipped = np.negative(flipped);
            using var negReversed = np.negative(reversed);

            negFlipped.strides.Should().Equal(8L, 16L, 48L);
            negReversed.strides.Should().Equal(96L, 32L, 8L);
            ((double)negFlipped.GetValue(0, 0, 0)).Should().Be(-4.0);
            ((double)negReversed.GetValue(0, 0, 0)).Should().Be(-9.0);
        }

        /// <summary>
        ///     The two-output ufuncs (np.divmod, np.modf, np.frexp) never take NumPy's trivial loop, so even an operand the
        ///     trivial loop would serve gets NpyIter's allocation: an F (2, 1, 3) float64 operand's frexp mantissas have
        ///     strides (8, 48, 16), where np.negative of it has (8, 16, 16). Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void TwoOutputUfuncs_AlwaysTakeTheIteratorsLayout()
        {
            using var x = np.asfortranarray(np.arange(1.0, 7.0).reshape(2, 1, 3));
            using var p = np.arange(1.0, 25.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });

            var (mantissa, exponent) = np.frexp(x);
            var (quotient, remainder) = np.divmod(p, (NDArray)5.0);
            using (mantissa)
            using (exponent)
            using (quotient)
            using (remainder)
            {
                mantissa.strides.Should().Equal(8L, 48L, 16L);
                exponent.strides.Should().Equal(4L, 24L, 8L);
                quotient.strides.Should().Equal(32L, 64L, 8L);
                remainder.strides.Should().Equal(32L, 64L, 8L);
            }
        }

        /// <summary>
        ///     NumPy zeroes every stride of a fresh zero-size allocation, so the result of any ufunc over an empty operand has
        ///     all-zero strides (and the C, F and writeable flags of every empty array). Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void EmptyResults_HaveAllZeroStrides()
        {
            using var e = np.zeros(new Shape(2, 0, 3));

            using var neg = np.negative(e);
            using var sum = e + 1.0;
            using var gt = e > 1.0;

            neg.strides.Should().Equal(0L, 0L, 0L);
            sum.strides.Should().Equal(0L, 0L, 0L);
            gt.strides.Should().Equal(0L, 0L, 0L);
            neg.flags.c_contiguous.Should().BeTrue();
            neg.flags.f_contiguous.Should().BeTrue();
            neg.flags.writeable.Should().BeTrue();
        }

        /// <summary>
        ///     The keepdims re-insertion gives the kept axis NumPy's stride (its place in the input's memory order, the
        ///     running stride without growth): <c>np.sum(F, axis=1, keepdims=True)</c> of an F (2, 3, 4) array has byte
        ///     strides (8, 16, 16), of a transposed one (32, 64, 8); <c>np.median(F, axis=1, keepdims=True)</c> is NumPy's
        ///     newaxis view of the F result, strides (8, 0, 16), owning no data. Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void KeepdimsResults_PlaceTheKeptAxisAsNumPyDoes()
        {
            using var f = np.asfortranarray(np.arange(1.0, 25.0).reshape(2, 3, 4));
            using var p = np.arange(1.0, 25.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });

            using var sumF = np.sum(f, axis: 1, keepdims: true);
            using var sumP = np.sum(p, axis: 1, keepdims: true);
            using var medF = np.median(f, axis: 1, keepdims: true);

            sumF.strides.Should().Equal(8L, 16L, 16L);
            sumP.strides.Should().Equal(32L, 64L, 8L);
            medF.strides.Should().Equal(8L, 0L, 16L);
            medF.flags.owndata.Should().BeFalse();
            ((double)medF.GetValue(1, 0, 3)).Should().Be(20.0);
        }

        /// <summary>
        ///     np.positive is a ufunc, so its result is a fresh, writeable array in the ufunc layout, not a clone of the
        ///     operand: a permuted operand keeps its memory order (32, 64, 8), a read-only owner's result is writeable, a
        ///     broadcast row's is C (32, 8) (the stride-0 axis abstains; a K-order copy would be F), an F (2, 1, 3) float64
        ///     operand takes the trivial loop's F (8, 16, 16), and an int32 one cast by <c>dtype=float64</c> takes
        ///     NpyIter's (8, 48, 16). Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void Positive_IsAFreshWriteableUfuncResult()
        {
            using var p = np.arange(24.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });
            using var ro = np.arange(12.0).reshape(3, 4);
            ro.setflags(write: false);
            using var row = np.broadcast_to(np.arange(4.0), new Shape(3, 4));
            using var f = np.asfortranarray(np.arange(6.0).reshape(2, 1, 3));
            using var i32f = np.asfortranarray(np.arange(6).astype(np.int32).reshape(2, 1, 3));

            using var ofP = np.positive(p);
            using var ofRo = np.positive(ro);
            using var ofRow = np.positive(row);
            using var ofF = np.positive(f);
            using var cast = np.positive(i32f, dtype: np.float64);

            ofP.strides.Should().Equal(32L, 64L, 8L);
            ofRo.flags.writeable.Should().BeTrue("a ufunc result never inherits the operand's read-only flag");
            ofRow.strides.Should().Equal(32L, 8L);
            ofRow.flags.writeable.Should().BeTrue();
            ofF.strides.Should().Equal(8L, 16L, 16L);
            cast.strides.Should().Equal(8L, 48L, 16L);
            ((double)cast.GetValue(1, 0, 2)).Should().Be(5.0);
        }

        /// <summary>
        ///     Functions NumPy builds from ufunc calls keep the ufunc layout: np.clip (a ufunc) of an F (2, 1, 3) array with
        ///     scalar bounds takes the trivial loop's plain F strides (8, 16, 16), not the iterator's (8, 48, 16); np.sinc
        ///     (pi * x, np.where, sin, /) and np.i0 (piecewise over abs(x)) of a permuted operand keep its order
        ///     (32, 64, 8), though NumSharp computes each in one fused np.evaluate pass. The values are those of the same
        ///     call on a C-ordered copy. Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void UfuncCompositions_KeepTheUfuncLayout()
        {
            using var f213 = np.asfortranarray(np.arange(6.0).reshape(2, 1, 3));
            using var p = np.arange(1.0, 25.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });
            using var pc = np.ascontiguousarray(p);

            using var clipped = np.clip(f213, 1, 4);
            using var sinc = np.sinc(p);
            using var i0 = np.i0(p);
            using var sincC = np.sinc(pc);
            using var i0C = np.i0(pc);

            clipped.strides.Should().Equal(8L, 16L, 16L);
            sinc.strides.Should().Equal(32L, 64L, 8L);
            i0.strides.Should().Equal(32L, 64L, 8L);
            np.array_equal(sinc, sincC).Should().BeTrue("relabelling moves no value");
            np.array_equal(i0, i0C).Should().BeTrue();
        }

        /// <summary>
        ///     np.cumsum / np.cumprod of a bool operand run the int64 loop over the copy NumPy's accumulate iterator makes
        ///     in the order the ORIGINAL operand votes: a broadcast (3, 4) bool row's stride-0 axis abstains, so the
        ///     result is C (32, 8). Converting first with a K-order astype gave F (8, 24), the stride-0 axis sorted
        ///     innermost (the matrix tier's matrix.cumprod/bcast/bool cases). Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void BoolScans_OfABroadcast_VoteWithTheOriginalOperand()
        {
            using var row = np.broadcast_to(np.array(new[] { true, false, true, true }), new Shape(3, 4));

            using var sum0 = np.cumsum(row, axis: 0);
            using var prod0 = np.cumprod(row, axis: 0);
            using var sum1 = np.cumsum(row, axis: 1);

            sum0.strides.Should().Equal(32L, 8L);
            prod0.strides.Should().Equal(32L, 8L);
            sum1.strides.Should().Equal(32L, 8L);
            sum0.ToArray<long>().Should().Equal(1L, 0L, 1L, 1L, 2L, 0L, 2L, 2L, 3L, 0L, 3L, 3L);
        }

        /// <summary>
        ///     The logical reductions place a kept axis as NumPy does too: <c>np.any(f, axis=0, keepdims=True)</c> of an F
        ///     (2, 3, 4) bool array has strides (1, 1, 3), along axis 1 (1, 2, 2); a tuple axis places every kept axis over
        ///     the original operand (<c>axis=(0, 2)</c>: (1, 1, 3)); and an empty keepdims result has all-zero strides, as
        ///     every fresh zero-size allocation does (<c>np.sum(np.zeros((0, 3)), axis=1, keepdims=True)</c>: (0, 0)),
        ///     while a reduced axis of length 0 leaves the running stride on the kept axis
        ///     (<c>np.sum(np.zeros((2, 0, 3)), axis=1, keepdims=True)</c>: (24, 24, 8)). Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void KeepdimsResults_LogicalAndEmpty_FollowNumPy()
        {
            using var f = np.asfortranarray(np.equal(np.mod(np.arange(24).reshape(2, 3, 4), 3), 0));
            using var e03 = np.zeros(new Shape(0, 3));
            using var e203 = np.zeros(new Shape(2, 0, 3));

            using var any0 = np.any(f, axis: 0, keepdims: true);
            using var all1 = np.all(f, axis: 1, keepdims: true);
            using var any02 = np.any(f, axis: new[] { 0, 2 }, keepdims: true);
            using var sumEmpty = np.sum(e03, axis: 1, keepdims: true);
            using var maxEmpty = np.max(e03, axis: 1, keepdims: true);
            using var sumOverEmptyAxis = np.sum(e203, axis: 1, keepdims: true);

            any0.strides.Should().Equal(1L, 1L, 3L);
            all1.strides.Should().Equal(1L, 2L, 2L);
            any02.strides.Should().Equal(1L, 1L, 3L);
            sumEmpty.strides.Should().Equal(0L, 0L);
            maxEmpty.strides.Should().Equal(0L, 0L);
            sumOverEmptyAxis.strides.Should().Equal(24L, 24L, 8L);
        }
    }
}
