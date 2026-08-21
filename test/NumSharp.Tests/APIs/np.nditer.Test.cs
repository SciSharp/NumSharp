using System;
using System.Globalization;
using System.Linq;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     Tests for <see cref="np.nditer"/>. Every expected value comes from running
    ///     NumPy 2.4.2 in the implementing session.
    /// </summary>
    [TestClass]
    public class np_nditer_Test
    {
        private static string V(NDArray x) => Convert.ToString(x.GetAtIndex(0), CultureInfo.InvariantCulture);

        private static string S(NDArray x)
            => "[" + string.Join(" ", Enumerable.Range(0, (int)x.size)
                .Select(i => Convert.ToString(x.GetAtIndex(i), CultureInfo.InvariantCulture))) + "]";

        private static string Values(np.NDIterator it) => string.Join(" ", it.Select(v => V(v[0])));

        private static NDArray A() => np.arange(6).reshape(2, 3);   // [[0,1,2],[3,4,5]]

        // ----------------------------------------------------------------- iteration order

        [TestMethod]
        public void NdIter_Basic()
        {
            using var it = np.nditer(A());
            Values(it).Should().Be("0 1 2 3 4 5");
        }

        [TestMethod]
        public void NdIter_OrderF()
        {
            // numpy: [0, 3, 1, 4, 2, 5]
            using var it = np.nditer(A(), order: 'F');
            Values(it).Should().Be("0 3 1 4 2 5");
        }

        [TestMethod]
        public void NdIter_FContiguousInput_KeepOrder_FollowsMemory()
        {
            // numpy order='K' over an F-contiguous array: 0 3 1 4 2 5
            using var it = np.nditer(np.asfortranarray(A()));
            Values(it).Should().Be("0 3 1 4 2 5");
        }

        [TestMethod]
        public void NdIter_FContiguousInput_OrderC_FollowsLogicalOrder()
        {
            using var it = np.nditer(np.asfortranarray(A()), order: 'C');
            Values(it).Should().Be("0 1 2 3 4 5");
        }

        [TestMethod]
        public void NdIter_ReversedView_KeepOrder_FollowsMemory()
        {
            // numpy: a[:, ::-1] with order='K' -> 0 1 2 3 4 5 (strides negated back to memory order)
            using var it = np.nditer(A()[":, ::-1"]);
            Values(it).Should().Be("0 1 2 3 4 5");
        }

        [TestMethod]
        public void NdIter_ReversedView_OrderC_FollowsLogicalOrder()
        {
            // numpy: a[:, ::-1] with order='C' -> 2 1 0 5 4 3
            using var it = np.nditer(A()[":, ::-1"], order: 'C');
            Values(it).Should().Be("2 1 0 5 4 3");
        }

        [TestMethod]
        public void NdIter_ZeroD()
        {
            using var it = np.nditer(np.array(5.0));
            Values(it).Should().Be("5");
        }

        // ----------------------------------------------------------------- index tracking

        [TestMethod]
        public void NdIter_MultiIndex()
        {
            using var it = np.nditer(A(), flags: new[] {"multi_index"});
            var seen = new System.Text.StringBuilder();
            while (!it.finished)
            {
                seen.Append("(" + string.Join(",", it.multi_index) + ")=" + V(it[0]) + " ");
                it.iternext();
            }

            seen.ToString().Trim().Should().Be("(0,0)=0 (0,1)=1 (0,2)=2 (1,0)=3 (1,1)=4 (1,2)=5");
        }

        [TestMethod]
        public void NdIter_CIndex()
        {
            using var it = np.nditer(A(), flags: new[] {"c_index"});
            var seen = new System.Text.StringBuilder();
            while (!it.finished)
            {
                seen.Append(it.index + "=" + V(it[0]) + " ");
                it.iternext();
            }

            seen.ToString().Trim().Should().Be("0=0 1=1 2=2 3=3 4=4 5=5");
        }

        [TestMethod]
        public void NdIter_FIndex()
        {
            // numpy: f_index -> 0=0 2=1 4=2 1=3 3=4 5=5
            using var it = np.nditer(A(), flags: new[] {"f_index"});
            var seen = new System.Text.StringBuilder();
            while (!it.finished)
            {
                seen.Append(it.index + "=" + V(it[0]) + " ");
                it.iternext();
            }

            seen.ToString().Trim().Should().Be("0=0 2=1 4=2 1=3 3=4 5=5");
        }

        // ----------------------------------------------------------------- shape / coalescing

        [TestMethod]
        public void NdIter_WithoutMultiIndex_CoalescesShape()
        {
            // numpy: np.nditer(a).shape == (6,) — the iterator may coalesce contiguous axes.
            using var it = np.nditer(A());
            it.shape.Should().Equal(6L);
            it.ndim.Should().Be(1);
            it.itersize.Should().Be(6);
        }

        [TestMethod]
        public void NdIter_WithMultiIndex_KeepsOriginalShape()
        {
            using var it = np.nditer(A(), flags: new[] {"multi_index"});
            it.shape.Should().Equal(2L, 3L);
            it.ndim.Should().Be(2);
            it.itersize.Should().Be(6);
        }

        // ----------------------------------------------------------------- external loop

        [TestMethod]
        public void NdIter_ExternalLoop_ContiguousIsOneChunk()
        {
            using var it = np.nditer(A(), flags: new[] {"external_loop"});
            string.Join(" | ", it.Select(v => S(v[0]))).Should().Be("[0 1 2 3 4 5]");
        }

        [TestMethod]
        public void NdIter_ExternalLoop_StridedSplitsPerRow()
        {
            // numpy: a[:, ::2] external_loop -> array([0,2]), array([3,5])
            using var it = np.nditer(A()[":, ::2"], flags: new[] {"external_loop"});
            string.Join(" | ", it.Select(v => S(v[0]))).Should().Be("[0 2] | [3 5]");
        }

        [TestMethod]
        public void NdIter_ExternalLoop_BufferedRespectsBuffersize()
        {
            using var it = np.nditer(A(), flags: new[] {"external_loop", "buffered"}, buffersize: 4);
            string.Join(" | ", it.Select(v => S(v[0]))).Should().Be("[0 1 2 3] | [4 5]");
        }

        [TestMethod]
        public void NdIter_EnableExternalLoop_AfterConstruction()
        {
            using var it = np.nditer(A());
            it.enable_external_loop();
            string.Join(" | ", it.Select(v => S(v[0]))).Should().Be("[0 1 2 3 4 5]");
        }

        // ----------------------------------------------------------------- multiple operands

        [TestMethod]
        public void NdIter_TwoOperands()
        {
            using var it = np.nditer(new[] {A(), A() * 10});
            string.Join(" ", it.Select(v => V(v[0]) + "/" + V(v[1])))
                .Should().Be("0/0 1/10 2/20 3/30 4/40 5/50");
        }

        [TestMethod]
        public void NdIter_BroadcastsOperands()
        {
            // numpy: nditer([[1,2,3], [[10],[20]]]) -> 1/10 2/10 3/10 1/20 2/20 3/20
            using var it = np.nditer(new[] {np.array(new[] {1, 2, 3}), np.array(new[,] {{10}, {20}})});
            string.Join(" ", it.Select(v => V(v[0]) + "/" + V(v[1])))
                .Should().Be("1/10 2/10 3/10 1/20 2/20 3/20");
        }

        [TestMethod]
        public void NdIter_Properties()
        {
            using var it = np.nditer(new[] {A(), A() * 10});
            it.nop.Should().Be(2);
            it.operands.Should().HaveCount(2);
            it.dtypes.Should().HaveCount(2);
            it.has_index.Should().BeFalse();
            it.has_multi_index.Should().BeFalse();
            it.has_delayed_bufalloc.Should().BeFalse();
            it.iterationneedsapi.Should().BeFalse();   // no Python runtime in NumSharp
        }

        [TestMethod]
        public void NdIter_ItViews()
        {
            using var it = np.nditer(new[] {A(), A() * 10});
            var views = it.itviews;
            views.Should().HaveCount(2);
            S(views[0]).Should().Be("[0 1 2 3 4 5]");
            S(views[1]).Should().Be("[0 10 20 30 40 50]");
        }

        // ----------------------------------------------------------------- allocation

        [TestMethod]
        public void NdIter_AllocatesNullOperand_AndWritesThrough()
        {
            using var it = np.nditer(
                new[] {A(), null},
                op_flags: new[] {new[] {"readonly"}, new[] {"writeonly", "allocate"}},
                op_dtypes: new[] {NPTypeCode.Int64, NPTypeCode.Int64});

            foreach (var v in it)
                v[1].SetAtIndex(Convert.ToInt64(v[0].GetAtIndex(0)) * 2, 0);

            S(it.operands[1]).Should().Be("[0 2 4 6 8 10]");
        }

        [TestMethod]
        public void NdIter_AllocateWithoutOpDtypes_InfersFromOtherOperands()
        {
            // numpy: np.nditer([a, None]).operands[1] is int64 with shape (2,3).
            using var it = np.nditer(new[] {A(), null});
            it.operands[1].typecode.Should().Be(NPTypeCode.Int64);
            it.operands[1].shape.Should().Equal(2L, 3L);
        }

        [TestMethod]
        public void NdIter_AllocateWithoutOpDtypes_PromotesAcrossOperands()
        {
            // numpy: np.nditer([int_a, float_b, None]).operands[2] is float64.
            using var it = np.nditer(new[] {A(), np.arange(3).astype(NPTypeCode.Double), null});
            it.operands[2].typecode.Should().Be(NPTypeCode.Double);
        }

        // ----------------------------------------------------------------- read-write

        [TestMethod]
        public void NdIter_ReadWrite_MutatesTheOperand()
        {
            var c = np.arange(3);
            using (var it = np.nditer(c, op_flags: new[] {"readwrite"}))
            {
                foreach (var v in it)
                    v[0].SetAtIndex(Convert.ToInt64(v[0].GetAtIndex(0)) * 2, 0);
            }

            S(c).Should().Be("[0 2 4]");
        }

        [TestMethod]
        public void NdIter_ReadWrite_Buffered_FlushesOnClose()
        {
            var c = np.arange(3).astype(NPTypeCode.Double);
            var it = np.nditer(c, op_flags: new[] {"readwrite"}, flags: new[] {"buffered"},
                op_dtypes: new[] {NPTypeCode.Double});

            foreach (var v in it)
                v[0].SetAtIndex(9.0, 0);

            it.close();
            S(c).Should().Be("[9 9 9]");
        }

        [TestMethod]
        public void NdIter_ReadWrite_TwoOperands()
        {
            var c = np.arange(3);
            var d = np.arange(3);
            using (var it = np.nditer(new[] {c, d},
                op_flags: new[] {new[] {"readwrite"}, new[] {"readwrite"}}))
            {
                foreach (var v in it)
                {
                    v[0].SetAtIndex(Convert.ToInt64(v[0].GetAtIndex(0)) + 1, 0);
                    v[1].SetAtIndex(Convert.ToInt64(v[1].GetAtIndex(0)) + 100, 0);
                }
            }

            S(c).Should().Be("[1 2 3]");
            S(d).Should().Be("[100 101 102]");
        }

        // ----------------------------------------------------------------- cursor control

        [TestMethod]
        public void NdIter_IterRange_LimitsIteration()
        {
            using var it = np.nditer(A(), flags: new[] {"ranged"});
            it.iterrange = (2, 5);
            Values(it).Should().Be("2 3 4");
            it.iterrange.Should().Be((2L, 5L));
        }

        [TestMethod]
        public void NdIter_IterIndex_Seeks()
        {
            using var it = np.nditer(A(), flags: new[] {"c_index"});
            it.iterindex = 3;
            V(it[0]).Should().Be("3");
            it.index.Should().Be(3);
        }

        [TestMethod]
        public void NdIter_Reset_RewindsToStart()
        {
            using var it = np.nditer(A());
            foreach (var _ in it) { }
            it.reset();
            Values(it).Should().Be("0 1 2 3 4 5");
        }

        [TestMethod]
        public void NdIter_Copy_ContinuesFromTheCurrentPosition()
        {
            // numpy: after one next(), copy() yields the REMAINING 1..5 — __next__ publishes
            // the current value and then advances.
            using var it = np.nditer(A());
            it.MoveNext();
            using var j = it.copy();
            Values(j).Should().Be("1 2 3 4 5");
        }

        [TestMethod]
        public void NdIter_IsItsOwnIterator_SecondPassResumes()
        {
            using var it = np.nditer(A());
            string first = string.Join(" ", it.Take(2).Select(v => V(v[0])));
            first.Should().Be("0 1");
            Values(it).Should().Be("2 3 4 5");
        }

        [TestMethod]
        public void NdIter_ForeachDoesNotCloseTheIterator()
        {
            // foreach disposes the enumerator it obtains; that must NOT free the iterator state,
            // or every property read after a loop would throw.
            using var it = np.nditer(A());
            foreach (var _ in it) { }

            it.finished.Should().BeTrue();
            it.itersize.Should().Be(6);
            it.nop.Should().Be(1);
        }

        [TestMethod]
        public void NdIter_RemoveAxis()
        {
            using var it = np.nditer(A(), flags: new[] {"multi_index"});
            it.remove_axis(0);
            it.shape.Should().Equal(3L);
            Values(it).Should().Be("0 1 2");
        }

        [TestMethod]
        public void NdIter_RemoveMultiIndex_EnablesCoalescing()
        {
            using var it = np.nditer(A(), flags: new[] {"multi_index"});
            it.remove_multi_index();
            it.shape.Should().Equal(6L);
            it.has_multi_index.Should().BeFalse();
        }

        // ----------------------------------------------------------------- buffering / casting

        [TestMethod]
        public void NdIter_BufferedCast()
        {
            using var it = np.nditer(A(), flags: new[] {"buffered"}, op_dtypes: new[] {NPTypeCode.Double});
            Values(it).Should().Be("0 1 2 3 4 5");
            it.dtypes[0].Should().Be(NPTypeCode.Double);
        }

        // ----------------------------------------------------------------- validation (verbatim NumPy texts)

        [TestMethod]
        public void NdIter_UnknownGlobalFlag_Throws()
            => new Action(() => np.nditer(A(), flags: new[] {"bogus"}))
                .Should().Throw<ArgumentException>()
                .WithMessage("Unexpected iterator global flag \"bogus\"");

        [TestMethod]
        public void NdIter_UnknownPerOpFlag_Throws()
            => new Action(() => np.nditer(A(), op_flags: new[] {"bogus"}))
                .Should().Throw<ArgumentException>()
                .WithMessage("Unexpected per-op iterator flag \"bogus\"");

        [TestMethod]
        public void NdIter_BadOrder_Throws()
            => new Action(() => np.nditer(A(), order: 'Q'))
                .Should().Throw<ArgumentException>()
                .WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");

        [TestMethod]
        public void NdIter_BadCasting_Throws()
            => new Action(() => np.nditer(A(), casting: "bogus"))
                .Should().Throw<ArgumentException>()
                .WithMessage("casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got 'bogus')");

        [TestMethod]
        public void NdIter_ExternalLoopWithMultiIndex_Throws()
            => new Action(() => np.nditer(A(), flags: new[] {"multi_index", "external_loop"}))
                .Should().Throw<ArgumentException>()
                .WithMessage("Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");

        [TestMethod]
        public void NdIter_ExternalLoopWithCIndex_Throws()
            => new Action(() => np.nditer(A(), flags: new[] {"c_index", "external_loop"}))
                .Should().Throw<ArgumentException>()
                .WithMessage("Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");

        [TestMethod]
        public void NdIter_ZeroSizedOperand_Throws()
            => new Action(() => np.nditer(np.zeros(new Shape(0, 3))))
                .Should().Throw<ArgumentException>()
                .WithMessage("Iteration of zero-sized operands is not enabled");

        [TestMethod]
        public void NdIter_ZeroSizedOperand_AllowedWithZeroSizeOk()
        {
            using var it = np.nditer(np.zeros(new Shape(0, 3)), flags: new[] {"zerosize_ok"});
            it.Should().BeEmpty();
        }

        [TestMethod]
        public void NdIter_NoOperands_Throws()
            => new Action(() => np.nditer(Array.Empty<NDArray>()))
                .Should().Throw<ArgumentException>()
                .WithMessage("Must provide at least one operand");

        [TestMethod]
        public void NdIter_GrowInnerAlias_IsAccepted()
        {
            // NumPy accepts both spellings; documented as grow_inner, originally growinner.
            using var a = np.nditer(A(), flags: new[] {"grow_inner"});
            using var b = np.nditer(A(), flags: new[] {"growinner"});
            Values(a).Should().Be(Values(b));
        }

        [TestMethod]
        public void NdIter_UseAfterClose_Throws()
        {
            var it = np.nditer(A());
            it.close();
            new Action(() => _ = it.nop).Should().Throw<InvalidOperationException>();
        }

        [TestMethod]
        public void NdIter_CloseIsIdempotent()
        {
            var it = np.nditer(A());
            it.close();
            new Action(() => it.close()).Should().NotThrow();
        }
    }
}
