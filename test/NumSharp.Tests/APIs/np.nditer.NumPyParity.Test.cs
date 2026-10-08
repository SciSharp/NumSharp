using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     np.nditer's NumPy 2.4.2 contracts that the open-bugs rundown's bug 6 (np.nditer over strings) and its neighbours
    ///     fixed, one test per contract, each expected value probed against NumPy 2.4.2. The oracle tiers
    ///     (<c>nditer_construct</c>, <c>nditer_protocol</c> and the string oracle's <c>striter</c> section) sweep the same
    ///     contracts over layouts and dtypes; these tests name them.
    /// </summary>
    [TestClass]
    public class np_nditer_NumPyParity_Test
    {
        /// <summary>Every operand-0 value of a walk, read with <see cref="NDArray.GetAtIndex"/>.</summary>
        /// <param name="it">The iterator, walked to its end.</param>
        /// <returns>The values in iteration order.</returns>
        private static List<object> Walk(np.NDIterator it)
        {
            var values = new List<object>();
            foreach (var v in it)
                values.Add(v[0].GetAtIndex(0));
            return values;
        }

        // ------------------------------------------------------------------ strings (bug 6)

        /// <summary>
        ///     A <c>U</c> array iterates like any dtype: <c>np.nditer(np.array(['ab', 'c']))</c> yields <c>'ab'</c>,
        ///     <c>'c'</c>, each a 0-d view of the array's own <c>&lt;U2</c> dtype.
        /// </summary>
        [TestMethod]
        public void StringArray_IteratesItsElements_InItsOwnDtype()
        {
            using var u = np.array(new[] { "ab", "c" });
            using var it = np.nditer(u);
            it.dtypes.Single().ToString(true).Should().Be("dtype('<U2')");
            Walk(it).Should().Equal("ab", "c");
        }

        /// <summary>A <c>readwrite</c> string view writes through: every element becomes <c>'zz'</c>.</summary>
        [TestMethod]
        public void StringArray_ReadWriteViews_WriteThrough()
        {
            using var u = np.array(new[] { "ab", "c" });
            using (var it = np.nditer(u, op_flags: new[] { "readwrite" }))
            {
                foreach (var v in it)
                {
                    using var zz = np.array("zz");
                    np.copyto(v[0], zz);
                }
            }

            u.GetAtIndex(0).Should().Be("zz");
            u.GetAtIndex(1).Should().Be("zz");
        }

        /// <summary>
        ///     A buffered <c>op_dtypes=['S2']</c> request over a <c>U</c> array converts each element (ASCII), as NumPy's
        ///     buffered cast does: <c>b'ab'</c>, <c>b'cd'</c>.
        /// </summary>
        [TestMethod]
        public void StringArray_BufferedUnicodeToBytes_Converts()
        {
            using var u = np.array(new[] { "ab", "cd" });
            using var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("S2") }, casting: "unsafe");
            Walk(it).Select(o => System.Text.Encoding.ASCII.GetString((byte[])o)).Should().Equal("ab", "cd");
        }

        // ------------------------------------------------------------------ casts and copy routes

        /// <summary>
        ///     A cast operand needs a copy or a buffer: <c>np.nditer(int32_array, op_dtypes=['float64'])</c> is NumPy's
        ///     TypeError "Iterator operand required copying or buffering, but neither copying nor buffering was enabled".
        /// </summary>
        [TestMethod]
        public void CastWithoutCopyOrBuffer_IsNumPysTypeError()
        {
            using var a = np.arange(3).astype(np.int32);
            var act = () => np.nditer(a, op_dtypes: new[] { np.float64 });
            act.Should().Throw<TypeError>()
               .WithMessage("Iterator operand required copying or buffering, but neither copying nor buffering was enabled");
        }

        /// <summary>The <c>copy</c> route casts a read-only operand into a temporary: the walk reads 0.0, 1.0, 2.0.</summary>
        [TestMethod]
        public void CopyRoute_IteratesTheCastTemporary()
        {
            using var a = np.arange(3).astype(np.int32);
            using var it = np.nditer(a, op_flags: new[] { "readonly", "copy" }, op_dtypes: new[] { np.float64 });
            Walk(it).Should().Equal(0.0, 1.0, 2.0);
        }

        /// <summary>
        ///     <c>updateifcopy</c> writes the cast temporary back at <c>close()</c> (unsafe casting: 2.5 → 2), not before:
        ///     inside the iterator's scope the array still reads <c>[0, 1, 2]</c>, after it <c>[0, 2, 5]</c>.
        /// </summary>
        [TestMethod]
        public void UpdateIfCopy_WritesBackAtClose()
        {
            using var a = np.arange(3).astype(np.int32);
            var it = np.nditer(a, op_flags: new[] { "readwrite", "updateifcopy" }, op_dtypes: new[] { np.float64 },
                casting: "unsafe");
            foreach (var v in it)
            {
                using var scaled = np.multiply(v[0], 2.5);
                np.copyto(v[0], scaled);
            }

            a.ToArray<int>().Should().Equal(0, 1, 2);
            it.close();
            a.ToArray<int>().Should().Equal(0, 2, 5);
        }

        /// <summary>
        ///     A ONE-element buffered readwrite cast is written back when the walk ends, before <c>close()</c> (NumPy applies
        ///     its single-iteration shortcut to unbuffered iterators only): <c>[5]</c> becomes <c>[7]</c>.
        /// </summary>
        [TestMethod]
        public void OneElementBufferedWrite_ReachesTheArrayWhenTheWalkEnds()
        {
            using var a = np.array(new[] { 5 });
            using var it = np.nditer(a, new[] { "buffered" }, new[] { "readwrite" }, new[] { np.float64 }, casting: "unsafe");
            foreach (var v in it)
            {
                using var seven = np.array(7.5);
                np.copyto(v[0], seven);
            }

            a.ToArray<int>().Should().Equal(7);
        }

        // ------------------------------------------------------------------ the state contract

        /// <summary>
        ///     nditer_pywrap's started / finished flags: after the walk <c>finished</c> is true and the position getters raise
        ///     "Iterator is past the end"; <c>reset()</c> clears <c>finished</c> and the next pull yields element 0.
        /// </summary>
        [TestMethod]
        public void StateContract_FinishedThenReset()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a);
            Walk(it).Should().Equal(0L, 1L, 2L);
            it.finished.Should().BeTrue();
            var act = () => _ = it.iterindex;
            act.Should().Throw<ValueError>().WithMessage("Iterator is past the end");

            it.reset();
            it.finished.Should().BeFalse();
            it.MoveNext().Should().BeTrue();
            it.Current[0].GetAtIndex(0).Should().Be(0L);
        }

        /// <summary>
        ///     <c>delay_bufalloc</c>: the views refuse with NumPy's text until <c>reset()</c> allocates the buffers, and
        ///     <c>has_delayed_bufalloc</c> reads true, then false.
        /// </summary>
        [TestMethod]
        public void DelayedBufferAllocation_RefusesUntilReset()
        {
            using var a = np.arange(3);
            using var it = np.nditer(new NDArray[] { a, null }, new[] { "buffered", "delay_bufalloc" },
                new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } });
            using var output = it.operands[1];
            it.has_delayed_bufalloc.Should().BeTrue();
            var act = () => _ = it[0];
            act.Should().Throw<ValueError>()
               .WithMessage("Iterator construction used delayed buffer allocation, and no reset has been done yet");

            it.reset();
            it.has_delayed_bufalloc.Should().BeFalse();
            it[0].GetAtIndex(0).Should().Be(0L);
        }

        // ------------------------------------------------------------------ element views and setters

        /// <summary>A read-only operand's view is not writeable, a readwrite operand's view is (NumPy's write flags).</summary>
        [TestMethod]
        public void Views_CarryNumPysWriteFlags()
        {
            using var src = np.arange(3);
            using var dst = np.zeros(new Shape(3), np.int64);
            using var it = np.nditer(new[] { src, dst }, op_flags: new[] { new[] { "readonly" }, new[] { "readwrite" } });
            it[0].flags.writeable.Should().BeFalse();
            it[1].flags.writeable.Should().BeTrue();
        }

        /// <summary>
        ///     <c>it[i] = v</c> packs a Python int into the operand's dtype with NumPy's range check ("Python integer 300 out
        ///     of bounds for int8"), and refuses a read-only operand ("Iterator operand 0 is not writeable").
        /// </summary>
        [TestMethod]
        public void Setter_PacksWithNumPysRangeCheck_AndRefusesReadOnly()
        {
            using var src = np.arange(3);
            using var dst = np.zeros(new Shape(3), np.int8);
            using var it = np.nditer(new[] { src, dst }, op_flags: new[] { new[] { "readonly" }, new[] { "readwrite" } });
            using var big = NDArray.Scalar(300);
            var overflow = () => it[1] = big;
            overflow.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");

            using var one = NDArray.Scalar(1);
            var readOnly = () => it[0] = one;
            readOnly.Should().Throw<RuntimeError>().WithMessage("Iterator operand 0 is not writeable");
        }

        // ------------------------------------------------------------------ orders and layouts

        /// <summary>
        ///     <c>order='A'</c> is a FORCED order: Fortran when every operand is F-contiguous (an F array walks
        ///     <c>0 3 1 4 2 5</c>), C otherwise — a transposed 3-D view walks its logical C order under 'A', not its memory
        ///     order.
        /// </summary>
        [TestMethod]
        public void OrderA_IsFortranOnlyWhenEveryOperandIsFContiguous()
        {
            using var f = np.asfortranarray(np.arange(6).reshape(2, 3));
            using (var it = np.nditer(f, order: 'A'))
                Walk(it).Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);

            using var t = np.arange(24).reshape(2, 3, 4).transpose(new[] { 2, 0, 1 });
            using (var it = np.nditer(t, order: 'A'))
                Walk(it).Take(6).Should().Equal(0L, 4L, 8L, 12L, 16L, 20L);
        }

        /// <summary>
        ///     An allocated output is laid out in the iteration order (NumPy's npyiter_new_temp_array): next to an
        ///     F-contiguous input it is F-contiguous, not C.
        /// </summary>
        [TestMethod]
        public void AllocatedOutput_FollowsTheIterationOrder()
        {
            using var f = np.asfortranarray(np.arange(6).reshape(2, 3));
            using var it = np.nditer(new NDArray[] { f, null });
            using var output = it.operands[1];
            output.flags.f_contiguous.Should().BeTrue();
            output.flags.c_contiguous.Should().BeFalse();
        }

        /// <summary>
        ///     <c>remove_axis</c> over a reversed operand keeps the removed axis at its logical index 0: the walk of
        ///     <c>arange(12).reshape(3, 4)[::-1, ::-1]</c> without axis 1 is <c>3 7 11</c>, shape <c>(3,)</c>.
        /// </summary>
        [TestMethod]
        public void RemoveAxis_OnAReversedOperand_KeepsLogicalIndexZero()
        {
            using var b = np.arange(12).reshape(3, 4)["::-1, ::-1"];
            using var it = np.nditer(b, new[] { "multi_index" });
            it.remove_axis(1);
            it.shape.Should().Equal(3L);
            Walk(it).Should().Equal(3L, 7L, 11L);
        }

        /// <summary><c>enable_external_loop</c> resets: after two pulls, the chunks still start at the first run.</summary>
        [TestMethod]
        public void EnableExternalLoop_ResetsTheIterator()
        {
            using var b = np.arange(6).reshape(2, 3)[":, ::2"];
            using var it = np.nditer(b);
            it.MoveNext();
            it.MoveNext();
            it.enable_external_loop();
            var chunks = new List<long[]>();
            foreach (var v in it)
                chunks.Add(v[0].ToArray<long>());
            chunks.Should().HaveCount(2);
            chunks[0].Should().Equal(0L, 2L);
            chunks[1].Should().Equal(3L, 5L);
        }

        /// <summary>
        ///     np.nested_iters splits the flags and dtypes per level as NpyIter_NestedIters does: with <c>buffered</c> and
        ///     <c>op_dtypes=['float64']</c> the cast happens in the innermost level (outer dtypes <c>int32</c>, inner
        ///     <c>float64</c>), not as a construction error.
        /// </summary>
        [TestMethod]
        public void NestedIters_BufferedCast_HappensInTheInnermostLevel()
        {
            using var a = np.arange(6).reshape(2, 3).astype(np.int32);
            var its = np.nested_iters(a, new[] { new[] { 0 }, new[] { 1 } }, new[] { "buffered" },
                op_dtypes: new[] { np.float64 });
            try
            {
                its[0].dtypes.Single().Should().Be(np.int32);
                its[1].dtypes.Single().Should().Be(np.float64);
                var values = new List<double>();
                foreach (var _ in its[0])
                {
                    foreach (var y in its[1])
                        values.Add(y[0].GetDouble(0));
                }

                values.Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            }
            finally
            {
                foreach (var it in its)
                    it.close();
            }
        }

        // ------------------------------------------------------------------ the flat index

        /// <summary>
        ///     Without a multi-index the shape lists the iteration axes INNERMOST first and the iterator coalesces what the
        ///     index allows: a <c>c_index</c> walk of a C <c>(3, 4)</c> array is one axis <c>(12,)</c>, an <c>f_index</c> walk
        ///     cannot coalesce and reports <c>(4, 3)</c>.
        /// </summary>
        [TestMethod]
        public void IndexTracking_CoalescesWhatTheIndexAllows()
        {
            using var c = np.arange(12).reshape(3, 4);
            using (var it = np.nditer(c, new[] { "c_index" }))
                it.shape.Should().Equal(12L);
            using (var it = np.nditer(c, new[] { "f_index" }))
                it.shape.Should().Equal(4L, 3L);
        }

        /// <summary>
        ///     The flat index rides its own strides, so <c>remove_multi_index</c> keeps it right: an <c>f_index</c> over a C
        ///     <c>(3, 4)</c> array still counts <c>0 3 6 9 1 4 7 10 2 5 8 11</c> (it used to coalesce to one axis and count
        ///     <c>0 1 2 …</c>).
        /// </summary>
        [TestMethod]
        public void FIndex_AfterRemoveMultiIndex_StillCountsInFOrder()
        {
            using var c = np.arange(12).reshape(3, 4);
            using var it = np.nditer(c, new[] { "multi_index", "f_index" });
            it.remove_multi_index();
            it.shape.Should().Equal(4L, 3L);
            var indices = new List<long>();
            while (!it.finished)
            {
                indices.Add(it.index);
                it.iternext();
            }

            indices.Should().Equal(0L, 3L, 6L, 9L, 1L, 4L, 7L, 10L, 2L, 5L, 8L, 11L);
        }

        /// <summary>
        ///     <c>remove_multi_index</c> only coalesces (NpyIter_RemoveMultiIndex): a forced F walk of a C <c>(2, 3)</c> array
        ///     keeps walking in F order, <c>0 3 1 4 2 5</c>, shape <c>(2, 3)</c> (it used to re-sort the axes by stride and
        ///     walk C order).
        /// </summary>
        [TestMethod]
        public void RemoveMultiIndex_KeepsAForcedOrder()
        {
            using var c = np.arange(6).reshape(2, 3);
            using var it = np.nditer(c, new[] { "multi_index" }, order: 'F');
            it.remove_multi_index();
            it.shape.Should().Equal(2L, 3L);
            Walk(it).Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
        }

        /// <summary>
        ///     Setting <c>index</c> reads the index strides (NpyIter_GotoIndex), flips included: over
        ///     <c>arange(12).reshape(3, 4)[::-1]</c> with <c>f_index</c>, index 5 is iteration index 1, element 1.
        /// </summary>
        [TestMethod]
        public void GotoIndex_ReadsTheIndexStrides()
        {
            using var r = np.arange(12).reshape(3, 4)["::-1"];
            using var it = np.nditer(r, new[] { "f_index" });
            it.index = 5;
            it.iterindex.Should().Be(1L);
            it[0].GetAtIndex(0).Should().Be(1L);
        }

        // ------------------------------------------------------------------ reductions through op_axes

        /// <summary>
        ///     The op_axes reduction checks are NumPy's (npyiter_check_reduce_ok_and_set_flags), in its order and texts: a
        ///     write-only reduction output, and a reduction without <c>reduce_ok</c> naming the operand axis (-1 for a
        ///     newaxis entry).
        /// </summary>
        [TestMethod]
        public void OpAxesReduction_RaisesNumPysErrors()
        {
            using var x = np.arange(24).reshape(2, 3, 4);
            var writeOnly = () => np.nditer(new NDArray[] { x, null }, new[] { "reduce_ok" },
                new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, op_axes: new[] { null, new[] { 0, 1, -1 } });
            writeOnly.Should().Throw<ValueError>()
                     .WithMessage("output operand requires a reduction, but is flagged as write-only, not read-write");

            var noReduceOk = () => np.nditer(new NDArray[] { x, null }, null,
                new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } }, op_axes: new[] { null, new[] { 0, 1, -1 } });
            noReduceOk.Should().Throw<ValueError>()
                      .WithMessage("output operand requires a reduction along dimension -1, but the reduction is not enabled. " +
                                   "The dimension size of 1 does not match the expected output shape.");
        }

        /// <summary>
        ///     An allocated reduction output does not vote in the axis order: the external-loop chunks of a last-axis
        ///     reduction run along the reduced axis (six chunks of 4), and NumPy's unbuffered idiom sums it.
        /// </summary>
        [TestMethod]
        public void AllocatedReductionOutput_WalksTheReducedAxisInnermost()
        {
            using var x = np.arange(24).reshape(2, 3, 4);
            using (var it = np.nditer(new NDArray[] { x, null }, new[] { "reduce_ok", "external_loop" },
                       new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } }, op_axes: new[] { null, new[] { 0, 1, -1 } }))
            {
                using var output = it.operands[1];
                var lengths = new List<long>();
                foreach (var v in it)
                    lengths.Add(v[0].size);
                lengths.Should().Equal(4L, 4L, 4L, 4L, 4L, 4L);
            }

            using (var it = np.nditer(new NDArray[] { x, null }, new[] { "reduce_ok" },
                       new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } }, op_axes: new[] { null, new[] { 0, 1, -1 } }))
            {
                using var output = it.operands[1];
                np.copyto(output, 0, casting: "unsafe");
                foreach (var v in it)
                    np.add(v[1], v[0], @out: v[1]);
                output.flatten().ToArray<long>().Should().Equal(6L, 22L, 38L, 54L, 70L, 86L);
            }
        }

        /// <summary>
        ///     A construction that fails AFTER allocating an output — here NumPy's write-only reduction refusal, raised once
        ///     the output exists — releases the output it allocated: the buffer pool's takes and returns balance across the
        ///     failed call. The caller never received that array, so before the fix it held its pooled buffer until a
        ///     finalizer ran (found by the oracle's leak gate on the nditer_protocol tier's error cases).
        /// </summary>
        [TestMethod]
        public void FailedConstruction_ReleasesTheOutputItAllocated()
        {
            using var x = np.arange(24).reshape(2, 3, 4);

            void Attempt()
            {
                try
                {
                    using var it = np.nditer(new NDArray[] { x, null }, new[] { "reduce_ok" },
                        new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, op_axes: new[] { null, new[] { 0, 1, -1 } });
                }
                catch (ValueError)
                {
                    // NumPy's refusal is the point; the pool balance below is the contract.
                }
            }

            Attempt();   // warms the code paths and the pool's buckets
            for (int attempt = 0; attempt < 5; attempt++)
            {
                GcQuiescence.CollectYoung();
                var opened = GcQuiescence.OpenWindow();
                long taken = SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
                long returned = SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;
                Attempt();
                long escaped = (SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs - taken)
                             - (SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed - returned);
                // A collection inside the window could have run a finalizer that returned someone else's buffer.
                if (!GcQuiescence.Undisturbed(opened))
                    continue;

                escaped.Should().Be(0, "the allocated output must go back to the pool when the construction fails");
                return;
            }

            Assert.Inconclusive("every measurement window was disturbed by a garbage collection");
        }

        /// <summary>
        ///     With a GIVEN reduction output, a K-order walk of an F-ordered input follows the input's memory order (NumPy's
        ///     npyiter_find_best_axis_ordering treats the output's zero strides as ambiguous): <c>0 4 8 12 16 20 …</c>.
        /// </summary>
        [TestMethod]
        public void GivenReductionOutput_KOrderFollowsTheInputsMemoryOrder()
        {
            using var f = np.asfortranarray(np.arange(24).reshape(2, 3, 4));
            using var output = np.zeros(new Shape(2, 3), np.int64);
            using var it = np.nditer(new[] { f, output }, new[] { "reduce_ok" },
                new[] { new[] { "readonly" }, new[] { "readwrite" } }, op_axes: new[] { null, new[] { 0, 1, -1 } });
            Walk(it).Take(6).Should().Equal(0L, 4L, 8L, 12L, 16L, 20L);
        }
    }
}
