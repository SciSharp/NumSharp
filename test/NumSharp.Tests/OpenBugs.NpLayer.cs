using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — divergences the np-layer dtype plan found on the way
    // =====================================================================
    //
    // docs/plans/np-layer-dtype-logic.md moved each np.* function's dtype logic below its facade and fixed what it found
    // in the code it moved (D4: one small commit per divergence). Six divergences sat outside the moved code; on
    // 2026-10-31 the owner asked for them to be pinned and deferred ("Create an open bug reproducing the problem/s and
    // defer to tackle them later"). Each group below is one of them, reproduced through public calls with the smallest
    // input that shows it:
    //
    //   NL1  FIXED (the plan's D24, 2026-11-06): np.copyto into a StringDType array under where= converted the WHOLE
    //        source, masked-off elements included, so a ufunc with a StringDType out= and where= (np.logical_and / or /
    //        xor / not), whose temporary leaves its masked-off slots uninitialized, raised "invalid value encountered in
    //        bool to string cast" at random. Only the selected elements are converted now; its tests moved to
    //        StringDTypeMovementTests.Copyto_WhereMask_ConvertsOnlyTheSelectedElements.
    //   NL2  FIXED (the plan's D24, 2026-11-06): np.median of complex data holding a NaN returned the mean of the middle
    //        values, not NumPy's NaN element (_median_nancheck); its tests moved to np_median_BattleTests
    //        (Median_Complex_WithANaN_*), and the oracle's complex-median excuse is gone.
    //   NL3  MISALIGNED (the plan's D24, 2026-11-06: "Keep SIMD lanes"): the float32 / float64 np.prod multiplies in
    //        SIMD lane order, not NumPy's sequential order (an ULP apart; an overflow before a zero is NaN in NumPy, 0
    //        here), flat, along an axis and through out=; initial= or where= run NumPy's loop. The lanes are 3.7-16x
    //        NumPy; its cases moved to np_prod_lane_order_tests as [Misaligned] tests.
    //   NL4  FIXED (the plan's D24, 2026-11-06): 1-D np.polyval with complex coefficients and an infinite real x gave
    //        NaN+NaNj where NumPy keeps the coefficient's imaginary part (its first product is in x's dtype); its tests
    //        moved to PolynomialTests.Polyval_FirstProduct_IsInXsDtype.
    //   NL5  FIXED (the plan's D24, 2026-11-06): argument checks in another order than NumPy's - np.nested_iters ran its
    //        converters after the axes and operand checks (and raised its no-operand error as a plain ArgumentException),
    //        np.einsum checked its out= extent after the cast, and contracted an explicit path without planning it. Its
    //        tests moved to np_nested_iters_tests (NoOperand_IsNumPysValueError, Converters_RunBeforeTheAxesAndOperandChecks)
    //        and EinsumPathTests (Einsum_ExplicitPath_IsPlannedFirst, Einsum_OutExtentError_PrecedesTheCastError).
    //   NL6  FIXED by parity QA-45 (78b122b0): np.array of a ragged jagged C# array (int[][], double[][], ...) truncated or
    //        threw IndexOutOfRangeException where NumPy raises its inhomogeneous-shape ValueError; its test stays below as
    //        the regression pin, without [OpenBugs].
    //   NL7  FIXED by the woven subclass paths: the plan's B3, user code that np.apply_along_axis / np.apply_over_axes
    //        call back ran with the old subclass boundary's per-thread depth raised, so a numpy.matrix or chararray the
    //        callback passed to an np.* function came back plain. The woven functions hand plain VIEWS to their
    //        computations instead of raising a depth, so a callback is ordinary user code again; its tests stay below as
    //        the regression pins, without [OpenBugs].
    //   NL8  np.sum's add.reduce step `*acc += pairwise_sum(call)` leaves the NaN of a NaN + NaN to the JIT: a Debug
    //        build of NumSharp.Core keeps the call's NaN where NumPy (and a Release build, by the JIT's operand order)
    //        keeps the accumulator's. Found by the journey4 merge's Debug gate, where it fails
    //        ShortRows_SeveralNaNs_KeepTheRunningSumsNaN and Trapezoid_NaNSpacing_IsTheProductsFirstOperand.
    //   NL9  FIXED (the plan's D24, 2026-11-06): np.polyder differentiated p.size - 1 terms per order where NumPy's
    //        `n = len(p) - 1` takes the first axis, and computed m products where NumPy's recursion computes m + 1 (the last
    //        one discarded but still raising), so a 2-D p was refused and a 0-d p was not len()'s TypeError at every order.
    //        Its tests moved to PolynomialTests (Polyder_TwoDimensionalP_DifferentiatesAlongTheFirstAxis,
    //        Polyder_ZeroDimensionalP_IsLensTypeErrorAtEveryOrder, Polyder_DiscardedOrderZeroProduct_StillRaises).
    //   NL10 FIXED (the plan's D24, 2026-11-06): np.einsum_path with an ellipsis and too few operands leaked .NET's
    //        IndexOutOfRangeException where NumPy's _parse_einsum_input raises IndexError("list index out of range"), and
    //        an explicit path past the operands leaked ArgumentOutOfRangeException where NumPy's pop raises IndexError("pop
    //        index out of range"). Its tests moved to EinsumPathTests (EllipsisWithoutItsOperand_RaisesIndexError,
    //        ExplicitPathPastTheOperands_RaisesIndexError).
    //   NL11 FIXED (the plan's D24, 2026-11-06): np.isscalar answered False for a Half and a BigInteger (and a Char8 and a
    //        DateTime64), where NumPy's float16 scalar, Python int, bytes_ and datetime64 are scalars, and True for a Type
    //        object (typeof(int)), where NumPy's isscalar(int) is False. Its tests moved to np_isscalar_tests
    //        (ScalarStandIns_AreScalars, TypeObjects_AreNotScalars).
    //   NL12 FIXED by parity QA-30 (batch 13): np.broadcast_to's refusals were not NumPy's texts (a 1-tuple without its
    //        comma, one space before "and requested shape", the shape text for a source with more axes than the target), and
    //        np.mean / np.var / np.std with a where= mask that does not broadcast reached them through _count_reduce_items;
    //        its test stays below as the regression pin, without [OpenBugs], the more-axes refusal under the house type of a
    //        broadcast refusal (IncorrectShapeException, NumPy's ValueError) as the owner's QA-30 decision keeps it.
    //   NL13 FIXED (the plan's D24, 2026-11-06): np.max / np.min / np.amax / np.amin / np.nanmax / np.nanmin with a
    //        numpy.matrix operand computed initial= / where= and refused a matrix out= with the plain reduction's rank
    //        error, where NumPy calls the matrix method (matrix.max(axis, out)); the nan forms now take NumPy's subclass
    //        branch too. Its tests moved to MaxMinKeywordTests (MaxMinFamily_MatrixOperand_FollowsTheMatrixMethod,
    //        NanMaxNanMin_MatrixOperand_TakesNumPysSubclassBranch).
    //   NL14 np.einsum's out= rules are c_einsum's iterator rules minus its broadcast, whatever optimize= says: NumPy's
    //        iterator broadcasts an output label of extent 1 up to the out's extent (np.einsum('ij->ij', ones((1, 4)),
    //        out=empty((3, 4))) fills every row) where NumSharp refuses with its contraction-shape text; and with optimize=
    //        other than False NumPy plans first (einsum_path's parser texts for every parse error) and hands the out to the
    //        last contraction step (bmm_einsum / c_einsum), which accepts an int32 out of a float64 contraction under
    //        casting='safe' and refuses a misshapen out with "could not broadcast input array from shape (2,4) into shape
    //        (2,5)". NumSharp keeps the C parser's texts (linalg-einsum.md) and c_einsum's out checks for every optimize=;
    //        an explicit path is planned first since NL5. Found by NL5's fix (2026-11-06).
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call (probed on win-amd64). EXPECTED (when fixed):
    // each test passes; remove its [OpenBugs], move it next to its API's tests, and drop the oracle excuse named in its
    // summary, if any.
    // =====================================================================
    public partial class OpenBugs
    {
        // ------------------------------------------------------------------
        // NL12 — np.broadcast_to's error texts
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's broadcast_to refuses through its iterator, whose texts print a 1-tuple with its comma and put TWO
        ///     spaces before "and requested shape" (probed: <c>np.broadcast_to(np.ones(2), (3,))</c> is <c>operands could not
        ///     be broadcast together with remapped shapes [original->remapped]: (2,)  and requested shape (3,)</c>), and a
        ///     source with more axes than the target is <c>input operand has more dimensions than allowed by the axis
        ///     remapping</c> (a ValueError). np.mean with a mask that does not broadcast to the operand raises the same text,
        ///     because <c>_count_reduce_items</c> broadcasts the mask first.
        /// </summary>
        /// <remarks>Fixed by parity QA-30: <c>Shape.ValidateBroadcastTo</c> (src/NumSharp.Core/View/Shape.Broadcasting.cs)
        ///     raises NumPy's texts, the more-axes one under the house <see cref="IncorrectShapeException"/> of every
        ///     broadcast refusal. A regression pin now, without [OpenBugs]; the full matrix is
        ///     <c>Creation/np.broadcast.RefusalTexts.Test.cs</c>.</remarks>
        [TestMethod]
        public void NpLayer_BroadcastTo_ErrorTexts_ShouldBeNumPys()
        {
            new Action(() => np.broadcast_to(np.ones(new Shape(2)), new Shape(3))).Should().Throw<IncorrectShapeException>()
                .WithMessage("operands could not be broadcast together with remapped shapes [original->remapped]: (2,)  and requested shape (3,)");
            new Action(() => np.mean(np.arange(3.0), where: np.ones(new Shape(2), np.@bool))).Should().Throw<IncorrectShapeException>()
                .WithMessage("operands could not be broadcast together with remapped shapes [original->remapped]: (2,)  and requested shape (3,)");
            new Action(() => np.broadcast_to(np.ones(new Shape(2, 3)), new Shape(3))).Should().Throw<IncorrectShapeException>()
                .WithMessage("input operand has more dimensions than allowed by the axis remapping");
        }

        // ------------------------------------------------------------------
        // NL6 — np.array of a ragged jagged C# array
        // ------------------------------------------------------------------

        /// <summary>
        ///     FIXED by parity QA-45 (78b122b0, 2026-11-03): NumSharp gives NumPy's answer now, and this test pins it.
        ///     <c>np.array([[97], [99, 100]])</c> and its siblings: NumPy raises <c>ValueError: setting an array element with
        ///     a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) +
        ///     inhomogeneous part.</c> Before the fix NumSharp's typed path took the shape from the first element of each
        ///     level (<c>Shape.ExtractShape</c>), so a longer later row was truncated silently (<c>[[97], [99]]</c>, also for
        ///     double[][] and int[][][]) and a shorter one threw <see cref="IndexOutOfRangeException"/>; the typed overloads
        ///     now discover the shape first (<c>NDArray.DiscoverJaggedShape</c>).
        /// </summary>
        [TestMethod]
        public void NpLayer_Array_RaggedJaggedArray_ShouldRaiseTheInhomogeneousShapeError()
        {
            const string afterOne = "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.";
            const string afterTwo = "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (2, 2) + inhomogeneous part.";

            Action longerSecondRow = () => np.array(new int[][] { new[] { 97 }, new[] { 99, 100 } });
            Action shorterSecondRow = () => np.array(new int[][] { new[] { 97, 98 }, new[] { 99 } });
            Action doubles = () => np.array(new double[][] { new[] { 1.5 }, new[] { 2.5, 3.5 } });
            Action threeLevels = () => np.array(new int[][][] { new[] { new[] { 1 }, new[] { 2 } }, new[] { new[] { 3 }, new[] { 4, 5 } } });

            longerSecondRow.Should().Throw<ValueError>().WithMessage(afterOne);
            shorterSecondRow.Should().Throw<ValueError>().WithMessage(afterOne);
            doubles.Should().Throw<ValueError>().WithMessage(afterOne);
            threeLevels.Should().Throw<ValueError>().WithMessage(afterTwo);
        }

        // ------------------------------------------------------------------
        // NL7 — FIXED: user callbacks see NumPy's classes (the plan's B3)
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.apply_along_axis(f, 1, matrix)</c> where <c>f</c> calls <c>np.sum(other_matrix, 0)</c> and
        ///     <c>np.ravel(chararray)</c>: in NumPy the callback is ordinary user code, so it gets a 1 x 2 matrix
        ///     <c>[[4, 6]]</c> and a chararray, exactly as outside (and as NumSharp gives with a PLAIN outer operand).
        ///     Regression pin (np-layer NL7, the plan's Appendix B3): the old subclass boundary raised a per-thread depth
        ///     around the whole computation, the user's callback included, so the callback got a plain (2,) array and a
        ///     plain array. The woven functions hand their computations plain VIEWS instead of raising a depth, so the
        ///     callback is ordinary user code and gets NumPy's classes.
        /// </summary>
        [TestMethod]
        public void NpLayer_ApplyAlongAxis_MatrixOperand_CallbackShouldSeeNumPysClasses()
        {
            using var m = np.asmatrix(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            using var other = np.asmatrix(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            using var text = np.@char.array(new[] { "ab", "cd" });
            // What the callback saw on its LAST call (both rows see the same): the classes and the sum's shape and values.
            Type sumClass = null, ravelClass = null;
            long[] sumShape = null;
            double[] sumValues = null;

            NDArray RowSum(NDArray row)
            {
                using var s = np.sum(other, 0);
                using var r = np.ravel(text);
                sumClass = s.GetType();
                sumShape = s.shape;
                sumValues = s.ToArray<double>();
                ravelClass = r.GetType();
                return np.sum(row);
            }

            using var result = np.apply_along_axis(RowSum, 1, m);

            sumClass.Should().Be(typeof(NDMatrix));
            sumShape.Should().Equal(1L, 2L);
            sumValues.Should().Equal(4.0, 6.0);
            ravelClass.Should().Be(typeof(NDCharArray));
        }

        /// <summary>
        ///     <c>np.apply_over_axes(h, matrix, [0])</c> where <c>h</c> calls <c>np.sum(other_matrix, 0)</c>: NumPy's callback
        ///     gets a 1 x 2 matrix, and so does NumSharp's: regression pin (np-layer NL7), where the old boundary's raised
        ///     depth gave the callback a plain (2,) array (see
        ///     <see cref="NpLayer_ApplyAlongAxis_MatrixOperand_CallbackShouldSeeNumPysClasses"/>). The result itself agrees
        ///     (a plain (1, 3) array <c>[[5, 7, 9]]</c> in both).
        /// </summary>
        [TestMethod]
        public void NpLayer_ApplyOverAxes_MatrixOperand_CallbackShouldSeeNumPysClasses()
        {
            using var m = np.asmatrix(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }));
            using var other = np.asmatrix(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            Type sumClass = null;
            long[] sumShape = null;

            NDArray AxisSum(NDArray arr, int axis)
            {
                using var s = np.sum(other, 0);
                sumClass = s.GetType();
                sumShape = s.shape;
                return np.sum(arr, axis);
            }

            using var result = np.apply_over_axes(AxisSum, m, new[] { 0 });

            result.ToArray<double>().Should().Equal(5.0, 7.0, 9.0);
            sumClass.Should().Be(typeof(NDMatrix));
            sumShape.Should().Equal(1L, 2L);
        }

        // ------------------------------------------------------------------
        // NL8 — np.sum's reduce step leaves the NaN of NaN + NaN to the JIT
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.sum</c> of a run whose first element is a NaN and whose rest holds another NaN keeps the FIRST
        ///     element's NaN in NumPy: add.reduce seeds the accumulator with the first element, then runs
        ///     <c>io1 += pairwise_sum(rest)</c>, and x86's <c>addsd</c> keeps its first operand's NaN when both are NaN.
        ///     NumPy 2.4.2 (probed, win-amd64): <c>[nan(1), 2, nan(2)]</c> sums to payload 1 in float64 and float32, and
        ///     rows <c>[[nan(1), 2, nan(2)], [1, nan(3), nan(4)]]</c> along axis 1 to payloads 1 and 3.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Add.Exact.cs</c>'s <c>AccumulateDouble</c> / <c>AccumulateSingle</c> write
        ///     the step as C#'s <c>*acc = *acc + s</c>, and C# leaves the operand order of a commutative <c>+</c> to the JIT
        ///     (<c>kernels.md</c>: encode NaN priority explicitly). A Debug build of NumSharp.Core compiles the step without
        ///     optimization, loads <c>s</c> first and keeps the pairwise sum's NaN (payloads 2 and 4); a Release build
        ///     happens to keep the accumulator in its register, NumPy's order, so this test FAILS in a Debug build and
        ///     passes in Release, and so do <c>NDEvaluateStreamingTests.ShortRows_SeveralNaNs_KeepTheRunningSumsNaN</c> and
        ///     <c>NpTrapezoidTest.Trapezoid_NaNSpacing_IsTheProductsFirstOperand</c> (np.trapezoid's final sum), which
        ///     stay unmarked so CI's Release run keeps guarding NumPy's answer. The fix chooses the accumulator's NaN
        ///     explicitly in both steps (and in any other <c>acc + s</c> combine the reductions write in C#).
        /// </remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL8")]
        public void NpLayer_Sum_RunOfNaNs_ShouldKeepTheAccumulatorsNaN_InEveryBuild()
        {
            static double N64(long payload) => BitConverter.Int64BitsToDouble(0x7FF8000000000000 | payload);
            static float N32(int payload) => BitConverter.Int32BitsToSingle(0x7FC00000 | payload);
            static long Bits64(NDArray r, int i) => BitConverter.DoubleToInt64Bits(r.GetDouble(i));

            using var a = np.array(new[] { N64(1), 2.0, N64(2) });
            using var s = np.sum(a);
            BitConverter.DoubleToInt64Bits(s.GetDouble()).Should().Be(0x7FF8000000000001, "float64 run: the first element's NaN");

            using var x = np.array(new[,] { { N64(1), 2.0, N64(2) }, { 1.0, N64(3), N64(4) } });
            using var rows = np.sum(x, 1);
            Bits64(rows, 0).Should().Be(0x7FF8000000000001, "row 0 keeps the accumulator's NaN");
            Bits64(rows, 1).Should().Be(0x7FF8000000000003, "row 1 keeps the accumulator's NaN");

            using var f = np.array(new[] { N32(1), 2f, N32(2) });
            using var sf = np.sum(f);
            BitConverter.SingleToInt32Bits(sf.GetSingle()).Should().Be(0x7FC00001, "float32 run: the first element's NaN");
        }

        // ------------------------------------------------------------------
        // NL14 — np.einsum's out= rules
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's c_einsum runs one iterator over the operands and the out, which broadcasts an output label of extent 1
        ///     up to the out's extent: <c>np.einsum('ij-&gt;ij', np.ones((1, 4)), out=np.zeros((3, 4)))</c> fills all three
        ///     rows with ones, and <c>np.einsum('ij-&gt;i', np.arange(4.).reshape(1, 4), out=np.zeros(3))</c> is
        ///     <c>[6, 6, 6]</c> (probed). NumSharp refuses both: "einsum() output parameter has shape (3,4) but the
        ///     contraction produces (1,4)".
        /// </summary>
        /// <remarks>Root cause: <c>DefaultEngine.Einsum</c> (src/NumSharp.Core/Backends/Default/LinearAlgebra/
        ///     Default.Einsum.cs) compares the out's shape with the contraction's and never broadcasts the result into
        ///     it.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL14")]
        public void NpLayer_Einsum_OutBroadcastsALengthOneOutputLabel()
        {
            using var rows = np.zeros(new Shape(3, 4));
            np.einsum("ij->ij", new[] { np.ones(new Shape(1, 4)) }, @out: rows);
            rows.ToArray<double>().Should().AllBeEquivalentTo(1.0);

            using var sums = np.zeros(new Shape(3));
            np.einsum("ij->i", new[] { np.arange(4.0).reshape(1, 4) }, @out: sums);
            sums.ToArray<double>().Should().Equal(6.0, 6.0, 6.0);
        }

        /// <summary>
        ///     With <c>optimize=True</c> NumPy's einsum runs <c>einsum_path</c> first and hands the out to the last
        ///     contraction step (<c>bmm_einsum</c>), so <c>np.einsum('ij,jk-&gt;ik', ones((2, 3)), ones((3, 4)),
        ///     out=zeros((2, 4), int32), optimize=True)</c> fills the out with 3 (probed) where the default optimize=False
        ///     raises the cast TypeError, and two operands for one term are einsum_path's "Number of einsum subscripts must be
        ///     equal to the number of operands." NumSharp raises the c_einsum TypeError and its C parser's text
        ///     ("fewer operands provided to einstein sum function than specified in the subscripts string") for every
        ///     optimize=.
        /// </summary>
        /// <remarks>Root cause: <c>DefaultEngine.Einsum</c> applies the c_einsum rules whatever optimize= says, and plans only
        ///     an explicit path (NL5).</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL14")]
        public void NpLayer_Einsum_OptimizeTrue_PlansFirstAndHandsTheOutToTheLastStep()
        {
            using var a = np.ones(new Shape(2, 3));
            using var b = np.ones(new Shape(3, 4));
            using var o = np.zeros(new Shape(2, 4), np.int32);

            np.einsum("ij,jk->ik", new[] { a, b }, @out: o, optimize: true);
            o.ToArray<int>().Should().AllBeEquivalentTo(3);

            new Action(() => np.einsum("ij->i", new[] { a, b }, optimize: true))
                .Should().Throw<ValueError>().WithMessage("Number of einsum subscripts must be equal to the number of operands.");
        }
    }
}
