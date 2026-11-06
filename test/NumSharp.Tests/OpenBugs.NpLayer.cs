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
    //   NL3  the flat np.prod of float32 / float64 multiplies in SIMD lane order, not NumPy's sequential order: values
    //        differ by an ULP, and a product that overflows before it meets a zero is NaN on one side and 0 on the other.
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
    //   NL7  the plan's B3: user code that np.apply_along_axis / np.apply_over_axes call back runs with the subclass
    //        boundary's depth raised, so a numpy.matrix or chararray the callback passes to an np.* function comes back
    //        plain. P7 would have removed the depth; the owner dropped P7 the same day ("Drop because np.* uses the lhs
    //        to pick which"), so B3 stays open and is pinned here with the six.
    //   NL8  np.sum's add.reduce step `*acc += pairwise_sum(call)` leaves the NaN of a NaN + NaN to the JIT: a Debug
    //        build of NumSharp.Core keeps the call's NaN where NumPy (and a Release build, by the JIT's operand order)
    //        keeps the accumulator's. Found by the journey4 merge's Debug gate, where it fails
    //        ShortRows_SeveralNaNs_KeepTheRunningSumsNaN and Trapezoid_NaNSpacing_IsTheProductsFirstOperand.
    //   NL9  np.polyder differentiates p.size - 1 terms per order where NumPy's `n = len(p) - 1` takes the first axis, and
    //        computes m products where NumPy's recursion computes m + 1 (the last one, at m == 0, discarded): a 2-D p is
    //        differentiated along its rows in NumPy and refused here, a 0-d p raises len()'s TypeError in NumPy at every
    //        order (0 included), and an N-D p whose extra product does not broadcast raises in NumPy, also at m == 0.
    //        Found by P8's move of polyder onto TensorEngine (434cc8d4), which kept the behaviour.
    //   NL10 FIXED (the plan's D24, 2026-11-06): np.einsum_path with an ellipsis and too few operands leaked .NET's
    //        IndexOutOfRangeException where NumPy's _parse_einsum_input raises IndexError("list index out of range"), and
    //        an explicit path past the operands leaked ArgumentOutOfRangeException where NumPy's pop raises IndexError("pop
    //        index out of range"). Its tests moved to EinsumPathTests (EllipsisWithoutItsOperand_RaisesIndexError,
    //        ExplicitPathPastTheOperands_RaisesIndexError).
    //   NL11 np.isscalar answers False for a Half (NumPy's float16 scalar) and a BigInteger (a Python int past int64),
    //        where NumPy answers True, and True for a Type object of a primitive (typeof(int)), where NumPy's isscalar(int)
    //        is False. Found by P8's snapshot tool, whose object operand (18835e6b) first ran np.isscalar over them; the
    //        move onto the array object (4cea01d6) kept the behaviour.
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
    //   NL3 also covers the axis products: an axis np.prod of float32 / float64 multiplies a row in SIMD lane order too
    //        (and so does the out= route, which computes the plain product); np.prod with initial= or a mask runs
    //        NumPy's sequential loop and is exact. Found by P10 item 1's reduce_kwargs cells (carved there).
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call (probed on win-amd64). EXPECTED (when fixed):
    // each test passes; remove its [OpenBugs], move it next to its API's tests, and drop the oracle excuse named in its
    // summary, if any.
    // =====================================================================
    public partial class OpenBugs
    {
        // ------------------------------------------------------------------
        // NL3 — the flat float32 / float64 np.prod multiplies in SIMD lane order
        // ------------------------------------------------------------------

        /// <summary>
        ///     The flat <c>np.prod</c> of a C-contiguous float32 (3, 4) array <c>[[2**64, 2**64, 1, 1], [0, 1, 1, 1],
        ///     [1, 1, 1, 1]]</c>: NumPy's <c>FLOAT_multiply</c> reduce is a sequential loop (<c>acc *= *src</c>,
        ///     loops_arithm_fp.dispatch.c.src: multiply is not pairwise), so <c>2**64 * 2**64</c> overflows to inf before
        ///     the zero and the product is NaN (0xFFC00000). NumSharp's flat route (<c>prod_elementwise_il</c>, the IL SIMD
        ///     reduction) multiplies lane by lane and combines the lanes at the end, so the zero enters a lane product before
        ///     the two 2**64 meet and the product is 0. Random float32 data differs in the last bits the same way: 18 of 35
        ///     C-contiguous 1-D products of 4 to 1,000 values. The oracle's unbounded "reduction summation/two-pass precision (algorithm
        ///     order)" excuse covers prod today; narrow it to sum with the fix.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float32Flat_ShouldMultiplyInNumPysSequentialOrder()
        {
            const float big = 18446744073709551616f;   // 2**64: its square overflows float32
            using var a = np.array(new float[,] { { big, big, 1, 1 }, { 0, 1, 1, 1 }, { 1, 1, 1, 1 } });
            using var r = np.prod(a);

            BitConverter.SingleToInt32Bits((float)r.GetValue()).Should().Be(unchecked((int)0xFFC00000));
        }

        /// <summary>
        ///     The flat <c>np.prod</c> of an F-ordered float32 (3, 4) array: NumPy's reduction iterator walks the operand in
        ///     MEMORY order (K order: the first axis is innermost here), so the product of these twelve values is 0x409D7A6B,
        ///     one ULP from the C-ordered array's 0x409D7A6C (which NumSharp matches). NumSharp gives 0x409D7A6D. The values
        ///     are the float32 casts of the fourth <c>uniform(0.5, 2.0, size=(3, 4))</c> draw of NumPy's
        ///     <c>default_rng(7)</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float32FOrdered_ShouldMultiplyInMemoryOrder()
        {
            using var c = np.array(new float[,]
            {
                { 1.0543044805526733f, 0.5056013464927673f, 1.7450716495513916f, 0.7316915988922119f },
                { 0.9013989567756653f, 1.8204982280731201f, 1.2646862268447876f, 1.7707253694534302f },
                { 1.4595757722854614f, 1.6126564741134644f, 0.637243390083313f, 1.3117157220840454f },
            });
            using var f = np.asfortranarray(c);
            using var r = np.prod(f);

            BitConverter.SingleToInt32Bits((float)r.GetValue()).Should().Be(0x409D7A6B);
        }

        /// <summary>
        ///     The flat <c>np.prod</c> of a float32 view reversed on both axes: NumPy's reductions pass
        ///     <c>NPY_ITER_DONT_NEGATE_STRIDES</c>, so a reversed view is multiplied in its LOGICAL order, where the two
        ///     2**64 come first and overflow before the zero: NaN (0xFFC00000). NumSharp gives 0.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float32Reversed_ShouldMultiplyInLogicalOrder()
        {
            const float big = 18446744073709551616f;   // 2**64
            using var owner = np.array(new float[,] { { 1, 1, 1, 1 }, { 1, 1, 1, 0 }, { 1, 1, big, big } });
            using var reversed = owner["::-1, ::-1"];
            using var r = np.prod(reversed);

            BitConverter.SingleToInt32Bits((float)r.GetValue()).Should().Be(unchecked((int)0xFFC00000));
        }

        /// <summary>
        ///     The flat <c>np.prod</c> of four float64 values: NumPy's sequential <c>DOUBLE_multiply</c> reduce gives NaN for
        ///     <c>[1e200, 1e200, 0, 1]</c> (inf, then inf * 0) and 0 for <c>[1e200, 0, 1e200, 1]</c>. NumSharp answers the
        ///     other way round (both answers fit lanes that pair the first value with the third and the second with the
        ///     fourth), so four values are enough to show the order; random float64 data differs in the last bits in 20 of
        ///     35 C-contiguous 1-D products of 4 to 1,000 values.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float64Flat_ShouldMultiplyInNumPysSequentialOrder()
        {
            using var overflowFirst = np.array(new[] { 1e200, 1e200, 0.0, 1.0 });
            using var zeroFirst = np.array(new[] { 1e200, 0.0, 1e200, 1.0 });
            using var r1 = np.prod(overflowFirst);
            using var r2 = np.prod(zeroFirst);

            BitConverter.DoubleToInt64Bits((double)r1.GetValue()).Should().Be(unchecked((long)0xFFF8000000000000));
            BitConverter.DoubleToInt64Bits((double)r2.GetValue()).Should().Be(0L);
        }

        /// <summary>
        ///     The AXIS <c>np.prod</c> of float64 rows: NumPy's reduce form over each row is the same sequential loop, so
        ///     <c>[[1e200, 1e200, 0, 1], [1e200, 0, 1e200, 1]]</c> along axis 1 is <c>[nan, 0]</c>; NumSharp's axis kernel
        ///     multiplies a row in lane order and answers <c>[0, nan]</c>. The <c>out=</c> route computes the same plain
        ///     product. With <c>initial=</c> or a mask the product runs NumPy's iterator and loop
        ///     (<c>NumPyMultiplyLoops</c>) and is exact, so the fix can route the plain product through the same loop.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float64Axis_ShouldMultiplyInNumPysSequentialOrder()
        {
            using var x = np.array(new[,] { { 1e200, 1e200, 0.0, 1.0 }, { 1e200, 0.0, 1e200, 1.0 } });
            using var r = np.prod(x, 1);
            using var o = np.zeros(new Shape(2));
            np.prod(x, 1, @out: o);

            BitConverter.DoubleToInt64Bits(r.GetAtIndex<double>(0)).Should().Be(unchecked((long)0xFFF8000000000000));
            BitConverter.DoubleToInt64Bits(r.GetAtIndex<double>(1)).Should().Be(0L);
            BitConverter.DoubleToInt64Bits(o.GetAtIndex<double>(0)).Should().Be(unchecked((long)0xFFF8000000000000));
        }

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
        // NL7 — the boundary depth reaches user callbacks (the plan's B3)
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.apply_along_axis(f, 1, matrix)</c> where <c>f</c> calls <c>np.sum(other_matrix, 0)</c> and
        ///     <c>np.ravel(chararray)</c>: in NumPy the callback is ordinary user code, so it gets a 1 x 2 matrix
        ///     <c>[[4, 6]]</c> and a chararray, exactly as outside (and as NumSharp gives with a PLAIN outer operand).
        ///     NumSharp's callback gets a plain (2,) array and a plain array: <c>NDMatrixBoundary.Wrapped</c> raises the
        ///     per-thread depth around the whole computation, the user's callback included, and every boundary rule reads
        ///     a raised depth as "nested library call, stay plain". The plan's Appendix B3; its fix was P7 (rules on
        ///     results, the depth deleted), which the owner dropped on 2026-10-31.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL7")]
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
        ///     gets a 1 x 2 matrix; NumSharp's gets a plain (2,) array, because the call runs inside the boundary's plain
        ///     region for the matrix operand (the depth raised, as in
        ///     <see cref="NpLayer_ApplyAlongAxis_MatrixOperand_CallbackShouldSeeNumPysClasses"/>). The result itself agrees
        ///     (a plain (1, 3) array <c>[[5, 7, 9]]</c> in both).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL7")]
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
        // NL9 — np.polyder counts p.size terms and computes one product fewer than NumPy
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's polyder is <c>n = len(p) - 1; y = p[:-1] * arange(n, 0, -1); val = p if m == 0 else
        ///     polyder(y, m - 1)</c>: the ramp has <c>len(p) - 1</c> terms (the FIRST axis), so a 2-D p is differentiated
        ///     along its rows, the rows broadcasting against the ramp. Probed: <c>np.polyder(np.ones((3, 2)))</c> is float64
        ///     <c>[[2., 1.], [2., 1.]]</c> and <c>np.polyder(np.arange(6.).reshape(3, 2))</c> is <c>[[0., 1.], [4., 3.]]</c>.
        ///     NumSharp's ramp has <c>p.size - 1</c> terms (5), which do not broadcast against the (2, 2) rows, so both calls
        ///     raise the multiply's IncorrectShapeException.
        /// </summary>
        /// <remarks>Root cause: <c>DefaultEngine.Polyder</c> (Backends/Default/Polynomial/Default.Polynomial.cs) takes
        ///     <c>n = p.size - 1</c> and loops m times instead of NumPy's len() and its recursion. A 1-D p is the same either
        ///     way, which is why the oracle's <c>poly</c> tier (1-D series) passes.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL9")]
        public void NpLayer_Polyder_TwoDimensionalP_ShouldDifferentiateAlongTheFirstAxis()
        {
            using var ones = np.ones(new Shape(3, 2));
            using var ramp = np.arange(6.0).reshape(3, 2);

            using var d1 = np.polyder(ones);
            d1.shape.Should().Equal(2L, 2L);
            d1.ToArray<double>().Should().Equal(2.0, 1.0, 2.0, 1.0);

            using var d2 = np.polyder(ramp);
            d2.ToArray<double>().Should().Equal(0.0, 1.0, 4.0, 3.0);
        }

        /// <summary>
        ///     NumPy's <c>n = len(p) - 1</c> runs before the <c>m == 0</c> test, so a 0-d p raises <c>TypeError: len() of
        ///     unsized object</c> at EVERY order, 0 included (probed for m = 0, 1 and 2). NumSharp returns the 0-d array for
        ///     m = 0 and raises the slice's IndexError (<c>too many indices for array</c>) for m &gt;= 1.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL9")]
        public void NpLayer_Polyder_ZeroDimensionalP_ShouldRaiseLensTypeErrorAtEveryOrder()
        {
            using var p = np.array(5.0);

            foreach (int m in new[] { 0, 1, 2 })
            {
                Action act = () => np.polyder(p, m);
                act.Should().Throw<TypeError>().WithMessage("len() of unsized object", $"m = {m}");
            }
        }

        /// <summary>
        ///     NumPy's recursion computes <c>y = p[:-1] * arange(n, 0, -1)</c> at every level, the last one (m == 0) included,
        ///     and discards that last product - but it still raises when the product does not broadcast. Probed:
        ///     <c>np.polyder(np.ones((3, 4)), 0)</c> raises <c>ValueError: operands could not be broadcast together with
        ///     shapes (2,4) (2,) </c>, and <c>np.polyder(np.ones((3, 2)), 2)</c> raises the same for <c>(0,2) (0,) </c> (the
        ///     third product, after two that succeed). NumSharp computes only m products: it returns p for m = 0, and its
        ///     m = 2 call meets its own p.size ramp error at the first product instead. NumSharp raises a broadcast mismatch
        ///     as IncorrectShapeException with NumPy's text (the house type for NumPy's ValueError there).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL9")]
        public void NpLayer_Polyder_ExtraProduct_ShouldRaiseWhereNumPysRecursionDoes()
        {
            using var wide = np.ones(new Shape(3, 4));
            using var tall = np.ones(new Shape(3, 2));

            Action orderZero = () => np.polyder(wide, 0);
            Action orderTwo = () => np.polyder(tall, 2);

            orderZero.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,4) (2,) ");
            orderTwo.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (0,2) (0,) ");
        }

        // ------------------------------------------------------------------
        // NL11 — np.isscalar: Half, BigInteger and Type objects
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's isscalar is <c>isinstance(element, generic) or type(element) in ScalarType or isinstance(element,
        ///     numbers.Number)</c>: a float16 scalar is a <c>generic</c>, so <c>np.isscalar(np.float16(1.5))</c> is True
        ///     (probed). <see cref="Half"/> is NumSharp's float16 scalar, but it is no CLR primitive, so NumSharp's
        ///     <c>type.IsPrimitive || obj is decimal</c> answers False.
        /// </summary>
        /// <remarks>Root cause: <c>NDArray.isscalar</c> (src/NumSharp.Core/Backends/NDArray.Numeric.cs) has no Half
        ///     case.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL11")]
        public void NpLayer_Isscalar_Half_ShouldBeAScalar()
        {
            np.isscalar((Half)1.5).Should().BeTrue("np.isscalar(np.float16(1.5)) is True");
        }

        /// <summary>
        ///     A Python int of any size is a <c>numbers.Number</c>, so <c>np.isscalar(2**70)</c> is True (probed).
        ///     <see cref="BigInteger"/> is how a C# caller spells a Python int past int64 (the house map), but it is no CLR
        ///     primitive, so NumSharp answers False.
        /// </summary>
        /// <remarks>Root cause: <c>NDArray.isscalar</c> has no BigInteger case.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL11")]
        public void NpLayer_Isscalar_BigInteger_ShouldBeAScalar()
        {
            np.isscalar(BigInteger.Pow(2, 70)).Should().BeTrue("np.isscalar(2**70) is True");
        }

        /// <summary>
        ///     A type is not a scalar: <c>np.isscalar(int)</c> and <c>np.isscalar(np.float64)</c> are False (probed; a class
        ///     object is neither a <c>generic</c> nor a <c>numbers.Number</c>). NumSharp reads a <see cref="Type"/> argument
        ///     as the type it names (<c>obj as Type ?? obj.GetType()</c>), so <c>np.isscalar(typeof(int))</c> answers for an
        ///     int: True.
        /// </summary>
        /// <remarks>Root cause: <c>NDArray.isscalar</c>'s <c>case Type _: break;</c>, which sends a Type object to the
        ///     primitive test of the type it names.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL11")]
        public void NpLayer_Isscalar_TypeObject_ShouldNotBeAScalar()
        {
            np.isscalar(typeof(int)).Should().BeFalse("np.isscalar(int) is False");
            np.isscalar(typeof(double)).Should().BeFalse("np.isscalar(float) is False");
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
