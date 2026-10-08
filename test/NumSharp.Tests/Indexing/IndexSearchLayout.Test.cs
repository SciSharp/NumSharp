using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     The objects the index searches and index transforms return, against NumPy 2.4.2 (the open-bugs rundown's bug 8
    ///     and the neighbours its sweep found). Every expected stride, flag and text below was probed on NumPy; the
    ///     layout-parity oracle (<c>layout_parity_oracle.jsonl</c>, families argwhere / flatnonzero / unravel / ravelmi)
    ///     covers the source-layout matrix, and these tests pin the contract's distinct rules, each one able to fail on
    ///     the code it replaced.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy builds these results as VIEWS: <c>np.flatnonzero</c> is the column of nonzero's <c>(k, 1)</c> buffer,
    ///     <c>np.argwhere</c> the transpose of the <c>(ndim, k)</c> stack <c>np.asarray</c> makes of nonzero's tuple, and
    ///     <c>np.unravel_index</c> the columns of one interleaved <c>indices.shape + (len(dims),)</c> buffer laid out along
    ///     the index array's iteration order. NumSharp returned separate owning C arrays (the pins of
    ///     <c>OpenBugs.IndexSearchLayout.cs</c>, which this class replaces). <c>np.ravel_multi_index</c> allocates through
    ///     NumPy's iterator, KEEPORDER over the coordinate arrays, where NumSharp always returned C.
    ///     </para>
    ///     <para>
    ///     Strides are asserted in BYTES (<see cref="NDArray.strides"/>, NumPy's convention). An empty result's strides are
    ///     not asserted: NumPy zeroes the strides of an empty array it allocates, a separate open divergence
    ///     (<c>OpenBugs.EmptyStrides.cs</c>).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class IndexSearchLayoutTests
    {
        // ---- the rundown's bug 8: the views NumPy returns ----------------------------------------------------------------

        /// <summary>
        ///     FIXED (bug 8): <c>np.flatnonzero(np.array([[0, 1, 2], [3, 0, 5]]))</c> is <c>[1, 2, 3, 5]</c>, the C-contiguous
        ///     column VIEW of nonzero's <c>(4, 1)</c> buffer (<c>flags.owndata</c> False). NumSharp's result owned its data.
        /// </summary>
        [TestMethod]
        public void Flatnonzero_IsAViewOfNonzerosBuffer()
        {
            using var a = np.array(new long[,] { { 0, 1, 2 }, { 3, 0, 5 } });
            using var r = np.flatnonzero(a);
            r.ToArray<long>().Should().Equal(1L, 2L, 3L, 5L);
            r.strides.Should().Equal(new long[] { 8 });
            r.flags.c_contiguous.Should().BeTrue();
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.owndata.Should().BeFalse("NumPy's flatnonzero is a column view of nonzero's index buffer");
            r.@base.shape.Should().Equal(new long[] { 4, 1 }, "the base is nonzero's (count, 1) multi-index array");
        }

        /// <summary>
        ///     FIXED (bug 8): <c>np.argwhere(np.array([[0, 1, 2], [3, 0, 5]]))</c> is <c>[[0, 1], [0, 2], [1, 0], [1, 2]]</c>
        ///     laid out as NumPy's transposed view: strides <c>(8, 32)</c> bytes, F-contiguous, not C-contiguous,
        ///     <c>flags.owndata</c> False, base the <c>(2, 4)</c> C stack. NumSharp's (4, 2) result was C (strides
        ///     <c>(16, 8)</c>) and owned its data.
        /// </summary>
        [TestMethod]
        public void Argwhere_IsNumPysTransposedView()
        {
            using var a = np.array(new long[,] { { 0, 1, 2 }, { 3, 0, 5 } });
            using var r = np.argwhere(a);
            r.ToArray<long>().Should().Equal(0L, 1L, 0L, 2L, 1L, 0L, 1L, 2L);
            r.strides.Should().Equal(new long[] { 8, 32 }, "NumPy's argwhere transposes the (ndim, k) index array");
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.c_contiguous.Should().BeFalse();
            r.flags.owndata.Should().BeFalse();
            var b = r.@base;
            b.shape.Should().Equal(new long[] { 2, 4 });
            b.strides.Should().Equal(new long[] { 32, 8 }, "the stack is C-ordered: row d holds coordinate d");
        }

        /// <summary>
        ///     FIXED (bug 8): <c>np.unravel_index(np.array([1, 5, 4]), (2, 3))</c> is <c>([0, 1, 1], [1, 2, 1])</c>, each a
        ///     VIEW of one interleaved <c>(3, 2)</c> int64 buffer (column <c>j</c> is <c>buffer[..., j]</c>): strides
        ///     <c>(16,)</c> bytes, not C-contiguous, <c>flags.owndata</c> False, one shared buffer. NumSharp's kernel wrote
        ///     each coordinate into its own C-contiguous owning array.
        /// </summary>
        [TestMethod]
        public void UnravelIndex_ReturnsViewsOfOneInterleavedBuffer()
        {
            using var idx = np.array(new long[] { 1, 5, 4 });
            var r = np.unravel_index(idx, new[] { 2, 3 });
            r[0].ToArray<long>().Should().Equal(0L, 1L, 1L);
            r[1].ToArray<long>().Should().Equal(1L, 2L, 1L);
            r[0].strides.Should().Equal(new long[] { 16 }, "NumPy's coordinates are columns of a (3, 2) buffer");
            r[0].flags.c_contiguous.Should().BeFalse();
            r[0].flags.owndata.Should().BeFalse();
            ReferenceEquals(r[0].Storage.InternalArray, r[1].Storage.InternalArray).Should().BeTrue("one shared buffer");
            r[1].Shape.offset.Should().Be(1, "coordinate j starts at element offset j of each row");
            r[0].@base.shape.Should().Equal(new long[] { 3, 2 });
        }

        /// <summary>
        ///     FIXED (bug 8): the empty forms keep the same ownership: <c>np.flatnonzero(np.zeros(0))</c> and
        ///     <c>np.argwhere(np.zeros((0, 3)))</c> report <c>flags.owndata</c> False in NumPy 2.4.2 — and so do the empty
        ///     string arrays whose index searches NumSharp answers from their shape.
        /// </summary>
        [TestMethod]
        public void EmptyIndexSearches_DoNotOwnTheirData()
        {
            using var f = np.zeros(new Shape(0L));
            using var flat = np.flatnonzero(f);
            flat.flags.owndata.Should().BeFalse("np.flatnonzero(np.zeros(0)).flags.owndata is False");
            using var e = np.zeros(new Shape(0L, 3L));
            using var where = np.argwhere(e);
            where.shape.Should().Equal(new long[] { 0, 2 }); // (k, ndim)
            where.flags.owndata.Should().BeFalse("np.argwhere(np.zeros((0, 3))).flags.owndata is False");
            using var text = np.zeros(new Shape(0L), "U2");
            using var textFlat = np.flatnonzero(text);
            textFlat.flags.owndata.Should().BeFalse("the shape-decided string answer runs the numeric path");
            var u = np.unravel_index(np.zeros(new Shape(0L, 3L), np.int64), new[] { 4, 6 });
            u[0].shape.Should().Equal(new long[] { 0, 3 });
            u[0].flags.owndata.Should().BeFalse("np.unravel_index's empty coordinates are views too");
        }

        // ---- argwhere: the 0-d case and the other ranks ----------------------------------------------------------------

        /// <summary>
        ///     A 0-d input is NumPy's <c>argwhere(atleast_1d(a))[:, :0]</c>: a <c>(1, 0)</c> VIEW of the promoted
        ///     <c>(1, 1)</c> result for a truthy element (strides <c>(8, 8)</c>), a <c>(0, 0)</c> one for a falsy element.
        ///     NumSharp returned a fresh owning array.
        /// </summary>
        [TestMethod]
        public void Argwhere_ZeroDim_IsAViewOfThePromotedResult()
        {
            using var truthy = np.argwhere(NDArray.Scalar(5L));
            truthy.shape.Should().Equal(new long[] { 1, 0 });
            truthy.strides.Should().Equal(new long[] { 8, 8 });
            truthy.flags.owndata.Should().BeFalse();
            truthy.@base.shape.Should().Equal(new long[] { 1, 1 });
            using var falsy = np.argwhere(NDArray.Scalar(0L));
            falsy.shape.Should().Equal(new long[] { 0, 0 });
            falsy.flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     The transposed stack at other ranks (NumPy 2.4.2): a 1-D input gives <c>(3, 1)</c> with strides
        ///     <c>(8, 24)</c>, C- AND F-contiguous (the extent-1 axis is ignored); a 3-D input <c>(16, 3)</c> with strides
        ///     <c>(8, 128)</c>, F only; a broadcast input reads its logical elements.
        /// </summary>
        [TestMethod]
        public void Argwhere_OtherRanks_AreTheTransposedStack()
        {
            using var r1 = np.argwhere(np.array(new long[] { 0, 1, 2, 0, 5 }));
            r1.ToArray<long>().Should().Equal(1L, 2L, 4L);
            r1.strides.Should().Equal(new long[] { 8, 24 });
            r1.flags.c_contiguous.Should().BeTrue();
            r1.flags.f_contiguous.Should().BeTrue();
            using var r3 = np.argwhere(np.arange(24).reshape(2, 3, 4) % 3);
            r3.shape.Should().Equal(new long[] { 16, 3 });
            r3.strides.Should().Equal(new long[] { 8, 128 });
            r3.flags.f_contiguous.Should().BeTrue();
            r3.flags.c_contiguous.Should().BeFalse();
            r3["2"].ToArray<long>().Should().Equal(0L, 1L, 0L);
            r3["15"].ToArray<long>().Should().Equal(1L, 2L, 3L);
            using var rb = np.argwhere(np.broadcast_to(np.array(new long[] { 0, 1, 2 }), new Shape(2, 3)));
            rb.ToArray<long>().Should().Equal(0L, 1L, 0L, 2L, 1L, 1L, 1L, 2L);
        }

        // ---- unravel_index: NumPy's iteration-order layout --------------------------------------------------------------

        /// <summary>
        ///     The interleaved buffer follows the index array's iteration order (NpyIter KEEPORDER, no stride negation):
        ///     an F-ordered index array gives coordinates with strides <c>(16, 32)</c>, a transposed 3-D one
        ///     <c>(16, 192, 64)</c>, a broadcast one C <c>(48, 16)</c>, a reversed 1-D one the natural <c>(16,)</c> with the
        ///     coordinates in LOGICAL order. NumSharp's coordinates were always C-ordered owning arrays.
        /// </summary>
        [TestMethod]
        public void UnravelIndex_LaysTheBufferOutAlongTheIterationOrder()
        {
            var f = np.unravel_index(np.asfortranarray(np.arange(6).reshape(2, 3)), new[] { 4, 6 });
            f[1].strides.Should().Equal(new long[] { 16, 32 });
            f[1].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            f[0].@base.strides.Should().Equal(new long[] { 16, 32, 8 });

            var t = np.unravel_index(np.arange(24).reshape(2, 3, 4).transpose(new[] { 2, 0, 1 }), new[] { 4, 6 });
            t[0].strides.Should().Equal(new long[] { 16, 192, 64 });

            var bc = np.unravel_index(np.broadcast_to(np.array(new long[] { 1, 2, 3 }), new Shape(2, 3)), new[] { 4, 6 });
            bc[1].strides.Should().Equal(new long[] { 48, 16 }, "a broadcast axis abstains from the sort; C order wins");
            bc[1].ToArray<long>().Should().Equal(1L, 2L, 3L, 1L, 2L, 3L);

            var rev = np.unravel_index(np.arange(6)["::-1"], new[] { 4, 6 });
            rev[1].strides.Should().Equal(new long[] { 16 }, "DONT_NEGATE_STRIDES: no axis is reversed");
            rev[1].ToArray<long>().Should().Equal(5L, 4L, 3L, 2L, 1L, 0L);

            var three = np.unravel_index(np.arange(6).reshape(2, 3), new[] { 2, 3, 4 });
            three[2].strides.Should().Equal(new long[] { 72, 24 });
            three[2].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 0L, 1L);
            three[2].Shape.offset.Should().Be(2);
        }

        /// <summary>
        ///     A non-int64 index array is cast into the iteration-order layout first: an F-ordered int16 one gives F-strided
        ///     coordinates (NumSharp's cast path laid it out in C order), a bool one reads 0 / 1, and a uint64 index at
        ///     2^63 wraps negative under NumPy's same_kind cast and is out of bounds with that value.
        /// </summary>
        [TestMethod]
        public void UnravelIndex_CastIndexArrays_KeepTheirIterationOrder()
        {
            var f16 = np.unravel_index(np.asfortranarray(np.arange(6).astype(np.int16).reshape(2, 3)), new[] { 4, 6 });
            f16[1].strides.Should().Equal(new long[] { 16, 32 });
            f16[1].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);

            var b = np.unravel_index(np.array(new[] { true, false, true }), new[] { 4, 6 }, 'F');
            b[0].ToArray<long>().Should().Equal(1L, 0L, 1L);

            Action wraps = () => np.unravel_index(np.array(new ulong[] { 1UL << 63, 1 }), new[] { 2, 3 });
            wraps.Should().Throw<ValueError>().WithMessage("index -9223372036854775808 is out of bounds for array with size 6");
        }

        /// <summary>
        ///     The coordinates share the buffer as NumPy's do: a write to one coordinate changes only that coordinate's
        ///     slots, and disposing one coordinate array leaves the others valid (each view holds a counted reference).
        /// </summary>
        [TestMethod]
        public void UnravelIndex_CoordinatesShareTheBuffer_AndOutliveEachOther()
        {
            var r = np.unravel_index(np.array(new long[] { 1, 5, 4 }), new[] { 2, 3 });
            r[0].SetAtIndex(99L, 0);
            r[0].ToArray<long>().Should().Equal(99L, 1L, 1L);
            r[1].ToArray<long>().Should().Equal(1L, 2L, 1L);
            r[0].@base.ToArray<long>().Should().Equal(99L, 1L, 1L, 2L, 1L, 1L);
            r[0].Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            r[1].ToArray<long>().Should().Equal(1L, 2L, 1L);
        }

        /// <summary>
        ///     A 0-d index array gives NumPy's scalars (<c>PyArray_Return</c>): fresh 0-d arrays that own their value and are
        ///     read-only (<c>flags.num</c> 263, a numpy scalar's flags).
        /// </summary>
        [TestMethod]
        public void UnravelIndex_ZeroDimIndex_ReturnsReadOnlyScalars()
        {
            var r = np.unravel_index(NDArray.Scalar(7L), new[] { 4, 6 });
            r.Should().HaveCount(2);
            foreach (var x in r)
            {
                x.ndim.Should().Be(0);
                x.flags.owndata.Should().BeTrue();
                x.flags.writeable.Should().BeFalse();
                x.flags.num.Should().Be(263);
            }
            r[0].GetInt64().Should().Be(1);
            r[1].GetInt64().Should().Be(1);
        }

        /// <summary>
        ///     <c>shape=()</c> is legal (NumPy 2.4.2): its one valid index is 0, which unravels to no coordinate; any other
        ///     index is out of bounds for size 1, and an index array of rank 1 or more is refused AFTER its indices are
        ///     checked (<c>multiple indices are not supported for 0d arrays</c>, gh-580). NumSharp refused the empty shape.
        /// </summary>
        [TestMethod]
        public void UnravelIndex_EmptyShape_IsNumPysUnitCase()
        {
            np.unravel_index(0L, new int[0]).Should().BeEmpty();
            np.unravel_index(NDArray.Scalar(0L), new int[0]).Should().BeEmpty();
            Action oob = () => np.unravel_index(1L, new int[0]);
            oob.Should().Throw<ValueError>().WithMessage("index 1 is out of bounds for array with size 1");
            Action multiple = () => np.unravel_index(np.array(new long[] { 0 }), new int[0]);
            multiple.Should().Throw<ValueError>().WithMessage("multiple indices are not supported for 0d arrays");
            Action empty = () => np.unravel_index(np.zeros(new Shape(0L, 2L), np.int64), new int[0]);
            empty.Should().Throw<ValueError>().WithMessage("multiple indices are not supported for 0d arrays");
            Action oobFirst = () => np.unravel_index(np.array(new long[] { 5 }), new int[0]);
            oobFirst.Should().Throw<ValueError>().WithMessage("index 5 is out of bounds for array with size 1");
        }

        /// <summary>
        ///     NumPy's statement order: a bad order letter (the argument parse) wins over everything, the dims overflow over
        ///     the index dtype, the index dtype's TypeError over an 'A' / 'K' order, and an 'A' / 'K' order over any index
        ///     value. NumSharp checked the order first for every letter.
        /// </summary>
        [TestMethod]
        public void UnravelIndex_ErrorsInNumPysOrder()
        {
            var big = new[] { 1 << 30, 1 << 30, 1 << 30 };
            Action qFloat = () => np.unravel_index(np.array(new[] { 1.5 }), new[] { 2, 3 }, 'Q');
            qFloat.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
            Action aFloat = () => np.unravel_index(np.array(new[] { 1.5 }), new[] { 2, 3 }, 'A');
            aFloat.Should().Throw<TypeError>().WithMessage("only int indices permitted");
            Action aOob = () => np.unravel_index(np.array(new long[] { 100 }), new[] { 2, 3 }, 'A');
            aOob.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            Action aOverflow = () => np.unravel_index(np.array(new long[] { 1 }), big, 'A');
            aOverflow.Should().Throw<ValueError>().WithMessage("dimensions are too large*");
            Action floatOverflow = () => np.unravel_index(np.array(new[] { 1.5 }), big);
            floatOverflow.Should().Throw<ValueError>().WithMessage("dimensions are too large*");
            Action kEmptyShape = () => np.unravel_index(0L, new int[0], 'k');
            kEmptyShape.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
        }

        /// <summary>
        ///     The out-of-range index reported is the FIRST one in NumPy's iteration order: <c>[1, 100, 2, 200][::-1]</c>
        ///     reports 200 (logical order, no stride negation) and the F-ordered <c>[[1, 100], [200, 2]]</c> reports 200 too
        ///     (memory order: 1, 200, 100, 2) where a C-order scan would meet 100 first.
        /// </summary>
        [TestMethod]
        public void UnravelIndex_ReportsTheFirstOutOfRangeIndexInIterationOrder()
        {
            Action rev = () => np.unravel_index(np.array(new long[] { 1, 100, 2, 200 })["::-1"], new[] { 2, 3 });
            rev.Should().Throw<ValueError>().WithMessage("index 200 is out of bounds for array with size 6");
            Action f = () => np.unravel_index(np.asfortranarray(np.array(new long[,] { { 1, 100 }, { 200, 2 } })), new[] { 2, 3 });
            f.Should().Throw<ValueError>().WithMessage("index 200 is out of bounds for array with size 6");
            Action c = () => np.unravel_index(np.array(new long[] { 1, -3, 100 }), new[] { 2, 3 });
            c.Should().Throw<ValueError>().WithMessage("index -3 is out of bounds for array with size 6");
        }

        /// <summary>
        ///     The reciprocal kernel (every dimension positive: a multiply-high and two shifts per peeled dimension, in
        ///     place of the hardware divide) is exact: against the scalar overload's plain C# division, over dimensions that
        ///     stress the reciprocal (1, 2, 3, powers of two and their neighbours, primes, 2^31 - 1) and indices at both ends
        ///     of the range and between, in both orders. A negative dimension takes the divmod kernel (C's truncating
        ///     division of a negative running quotient): <c>(-2, -3)</c> unravels index 5 to <c>(-1, 2)</c>, NumPy's answer.
        /// </summary>
        [TestMethod]
        public void UnravelIndex_ReciprocalKernel_IsExact()
        {
            int[] dimsPool = { 1, 2, 3, 5, 7, 10, 255, 256, 257, 1000, 65535, 65536, 999983, 1 << 30, int.MaxValue };
            var rng = new System.Random(8);
            foreach (int d0 in dimsPool)
            foreach (int d1 in dimsPool)
            foreach (int d2 in new[] { 1, 3, 4096, int.MaxValue })
            {
                var shape = new[] { d0, d1, d2 };
                long size = (long)d0 * d1;
                if (size > long.MaxValue / d2)
                    continue;
                size *= d2;
                var picks = new List<long> { 0, size - 1, size / 2, System.Math.Max(0, size - d2), System.Math.Min(size - 1, d2) };
                for (int r = 0; r < 4; r++)
                    picks.Add((long)(rng.NextDouble() * size) % size);
                var idx = np.array(picks.ToArray());
                foreach (char order in new[] { 'C', 'F' })
                {
                    var coords = np.unravel_index(idx, shape, order);
                    for (int i = 0; i < picks.Count; i++)
                    {
                        var want = np.unravel_index(picks[i], shape, order);
                        for (int k = 0; k < 3; k++)
                            Assert.AreEqual(want[k], coords[k].GetInt64(i), $"shape ({d0}, {d1}, {d2}) order {order} index {picks[i]} coordinate {k}");
                    }
                }
            }

            var neg = np.unravel_index(np.array(new long[] { 1, 5 }), new[] { -2, -3 });
            neg[0].ToArray<long>().Should().Equal(0L, -1L);
            neg[1].ToArray<long>().Should().Equal(1L, 2L);
        }

        /// <summary>NumPy's order converter accepts lowercase letters: <c>order='f'</c> is F order.</summary>
        [TestMethod]
        public void UnravelIndex_LowercaseOrder_IsAccepted()
        {
            var lower = np.unravel_index(np.array(new long[] { 1, 2 }), new[] { 2, 3 }, 'f');
            lower[0].ToArray<long>().Should().Equal(1L, 0L);
            lower[1].ToArray<long>().Should().Equal(0L, 1L);
            np.unravel_index(5L, new[] { 2, 3 }, 'c').Should().Equal(1L, 2L);
        }

        // ---- ravel_multi_index: the iterator's KEEPORDER output ---------------------------------------------------------

        /// <summary>
        ///     The result is NumPy's iterator ALLOCATE output, KEEPORDER over the coordinate arrays: an F pair gives an F
        ///     result (strides <c>(8, 16)</c>), a transposed pair <c>(8, 24)</c>, a mixed pair C (C wins conflicts), a
        ///     permuted 3-D pair <c>(8, 96, 32)</c>. NumSharp's result was always C.
        /// </summary>
        [TestMethod]
        public void RavelMultiIndex_AllocatesInKeepOrder()
        {
            var rows = np.array(new long[,] { { 0, 1, 1 }, { 1, 0, 1 } });
            var cols = np.array(new long[,] { { 1, 2, 0 }, { 0, 1, 2 } });
            var dims = new[] { 2, 3 };

            using var ff = np.ravel_multi_index(new[] { np.asfortranarray(rows), np.asfortranarray(cols) }, dims);
            ff.strides.Should().Equal(new long[] { 8, 16 });
            ff.ToArray<long>().Should().Equal(1L, 5L, 3L, 3L, 1L, 5L);

            using var tt = np.ravel_multi_index(new[] { rows.T, cols.T }, dims);
            tt.strides.Should().Equal(new long[] { 8, 24 });

            using var cf = np.ravel_multi_index(new[] { rows, np.asfortranarray(cols) }, dims);
            cf.strides.Should().Equal(new long[] { 24, 8 }, "conflicting operands: C order wins");

            using var t3 = np.ravel_multi_index(new[]
            {
                np.zeros(new Shape(2, 3, 4), np.int64).transpose(new[] { 2, 0, 1 }),
                np.ones(new Shape(2, 3, 4), np.int64).transpose(new[] { 2, 0, 1 }),
            }, dims);
            t3.strides.Should().Equal(new long[] { 8, 96, 32 });

            using var f32 = np.ravel_multi_index(new[] { np.asfortranarray(rows.astype(np.int32)), np.asfortranarray(cols.astype(np.int32)) }, dims);
            f32.strides.Should().Equal(new long[] { 8, 16 }, "cast coordinates keep the iterator's layout");

            // A 0-d int64 coordinate passes any stride test: it must be broadcast into the result's layout, never read in
            // place past its one element (NumPy: [3, 4, 5]).
            using var bc0 = np.ravel_multi_index(new[] { NDArray.Scalar(1L), np.array(new long[] { 0, 1, 2 }) }, dims);
            bc0.ToArray<long>().Should().Equal(3L, 4L, 5L);
            using var bcRow = np.ravel_multi_index(new[] { np.array(new long[,] { { 0 }, { 1 } }), np.array(new long[] { 0, 1, 2 }) }, dims);
            bcRow.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
        }

        /// <summary>
        ///     All-0-d coordinates give NumPy's scalar (read-only, owning); <c>dims=()</c> takes no coordinate array and gives
        ///     0 (NumPy's <c>np.int64(0)</c>; NumSharp refused the empty shape); a coordinate for it is NumPy's count error.
        /// </summary>
        [TestMethod]
        public void RavelMultiIndex_ScalarResults()
        {
            using var s = np.ravel_multi_index(new[] { NDArray.Scalar(1L), NDArray.Scalar(2L) }, new[] { 2, 3 });
            s.ndim.Should().Be(0);
            s.GetInt64().Should().Be(5);
            s.flags.writeable.Should().BeFalse("NumPy returns a numpy scalar");
            s.flags.owndata.Should().BeTrue();
            using var none = np.ravel_multi_index(new NDArray[0], new int[0]);
            none.ndim.Should().Be(0);
            none.GetInt64().Should().Be(0);
            np.ravel_multi_index(new long[0], new int[0]).Should().Be(0);
            Action one = () => np.ravel_multi_index(new[] { np.array(new long[] { 0 }) }, new int[0]);
            one.Should().Throw<ValueError>().WithMessage("parameter multi_index must be a sequence of length 0");
        }

        /// <summary>
        ///     NumPy refuses a zero dimension before folding any coordinate, in every mode (<c>cannot unravel if shape has
        ///     zero entries (is empty).</c>) — wrap used to divide by it — but not when there is nothing to fold.
        /// </summary>
        [TestMethod]
        public void RavelMultiIndex_ZeroDimension_RaisesInEveryMode()
        {
            foreach (var mode in new[] { "raise", "wrap", "clip" })
            {
                Action act = () => np.ravel_multi_index(new[] { np.array(new long[] { 0 }), np.array(new long[] { 0 }) }, new[] { 0, 3 }, mode);
                act.Should().Throw<ValueError>().WithMessage("cannot unravel if shape has zero entries (is empty).");
                Action scalar = () => np.ravel_multi_index(new long[] { 0, 0 }, new[] { 0, 3 }, mode);
                scalar.Should().Throw<ValueError>().WithMessage("cannot unravel if shape has zero entries (is empty).");
            }
            using var empty = np.ravel_multi_index(new[] { np.zeros(new Shape(0L), np.int64), np.zeros(new Shape(0L), np.int64) }, new[] { 0, 3 });
            empty.size.Should().Be(0);
        }

        /// <summary>
        ///     NumPy's statement order: the order letter (argument parse), the modes, the ravel strides ('A' / 'K', then the
        ///     dims overflow), the coordinate count, each coordinate's dtype, the broadcast (the iterator's text, every
        ///     shape listed), then the folding. NumSharp read the coordinates before the strides, and its errors were
        ///     ArgumentExceptions.
        /// </summary>
        [TestMethod]
        public void RavelMultiIndex_ErrorsInNumPysOrder()
        {
            var a = np.array(new long[,] { { 0, 1, 1 }, { 1, 0, 1 } });
            var b = np.array(new long[,] { { 1, 2, 0 }, { 0, 1, 2 } });
            var d = new[] { 2, 3 };
            var big = new[] { 1 << 30, 1 << 30, 1 << 30 };
            var z = np.array(new long[] { 0 });

            Action qMode = () => np.ravel_multi_index(new[] { a, b }, d, "bad", 'Q');
            qMode.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
            Action aMode = () => np.ravel_multi_index(new[] { a, b }, d, "bad", 'A');
            aMode.Should().Throw<ValueError>().WithMessage("clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bad')");
            Action aCount = () => np.ravel_multi_index(new[] { a }, d, "raise", 'A');
            aCount.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            Action overflowCount = () => np.ravel_multi_index(new[] { z }, big);
            overflowCount.Should().Throw<ValueError>().WithMessage("invalid dims: array size defined by dims is larger than the maximum possible size.");
            Action overflowA = () => np.ravel_multi_index(new[] { z, z, z }, big, "raise", 'K');
            overflowA.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            Action count = () => np.ravel_multi_index(new[] { a }, d);
            count.Should().Throw<ValueError>().WithMessage("parameter multi_index must be a sequence of length 2");
            Action fl = () => np.ravel_multi_index(new[] { a, b.astype(np.float64) }, d);
            fl.Should().Throw<TypeError>().WithMessage("only int indices permitted");
            Action invalid = () => np.ravel_multi_index(new[] { a, b + 5 }, d, "raise", 'c');
            invalid.Should().Throw<ValueError>().WithMessage("invalid entry in coordinates array");
            Action bcast = () => np.ravel_multi_index(new[] { np.zeros(new Shape(2, 3), np.int64), np.zeros(new Shape(4), np.int64) }, d);
            bcast.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,3) (4,) ");
            Action seqLen = () => np.ravel_multi_index(new[] { a, b }, d, new[] { "raise" }, 'A');
            seqLen.Should().Throw<ValueError>().WithMessage("list of clipmodes has wrong length (1 instead of 2)");
            Action scalarA = () => np.ravel_multi_index(new long[] { 1, 2 }, d, "raise", 'A');
            scalarA.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            Action scalarLen = () => np.ravel_multi_index(new long[] { 1 }, d);
            scalarLen.Should().Throw<ValueError>().WithMessage("parameter multi_index must be a sequence of length 2");
            Action scalarOob = () => np.ravel_multi_index(new long[] { 5, 2 }, d);
            scalarOob.Should().Throw<ValueError>().WithMessage("invalid entry in coordinates array");
            Action scalarQ = () => np.ravel_multi_index(new long[] { 1, 2 }, d, "raise", 'Q');
            scalarQ.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
            np.ravel_multi_index(new long[] { 1, 2 }, d, "raise", 'f').Should().Be(5, "(1, 2) in F order over (2, 3): 1 + 2 * 2");
        }

        // ---- the index-tuple generators -------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's generators return OWNING arrays where they build fresh ones (probed 2.4.2): <c>np.diag_indices</c>
        ///     repeats ONE fresh <c>np.arange</c> (the same object in every slot), <c>np.tril_indices</c> /
        ///     <c>np.triu_indices</c> (and their <c>_from</c> forms) return boolean-mask gathers. NumSharp returned aliases of
        ///     their own buffers (OWNDATA False). <c>np.mask_indices</c> is <c>np.nonzero</c>, so its entries ARE views of
        ///     one <c>(k, 2)</c> buffer (element stride 2), as NumPy's are.
        /// </summary>
        [TestMethod]
        public void IndexTupleGenerators_OwnWhatNumPyOwns()
        {
            var d = np.diag_indices(4);
            d[0].flags.owndata.Should().BeTrue("np.diag_indices(4)[0].flags.owndata is True");
            ReferenceEquals(d[0], d[1]).Should().BeTrue("NumPy's (idx,) * ndim repeats one object");
            np.diag_indices_from(np.arange(16).reshape(4, 4))[1].flags.owndata.Should().BeTrue();

            foreach (var r in new[] { np.tril_indices(4), np.triu_indices(4, 1), np.tril_indices(4, 1, 5),
                                      np.tril_indices_from(np.arange(16).reshape(4, 4)), np.triu_indices_from(np.arange(16).reshape(4, 4), 1) })
            {
                r[0].flags.owndata.Should().BeTrue();
                r[1].flags.owndata.Should().BeTrue();
                np.shares_memory(r[0], r[1]).Should().BeFalse();
            }
            np.tril_indices(4)[1].ToArray<long>().Should().Equal(0L, 0L, 1L, 0L, 1L, 2L, 0L, 1L, 2L, 3L);

            var m = np.mask_indices(3, np.triu);
            m[0].flags.owndata.Should().BeFalse("mask_indices is np.nonzero: columns of one (k, 2) buffer");
            m[0].strides.Should().Equal(new long[] { 16 });
            m[1].ToArray<long>().Should().Equal(0L, 1L, 2L, 1L, 2L, 2L);
        }

        // ---- the order converter everywhere ------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_OrderConverter</c> across the order-taking APIs (probed 2.4.2 on 21 of them): an unknown
        ///     letter is <c>order must be one of 'C', 'F', 'A', or 'K' (got 'Q')</c> (NumSharp's text lacked the ", or" and
        ///     carried a parameter suffix), a creation function refuses 'A' / 'K' with <c>only 'C' or 'F' order is
        ///     permitted</c> (NumSharp appended its own explanation), and every letter is accepted in lowercase (np.nditer
        ///     refused 'c' / 'f' / 'a' / 'k'). The types are <see cref="ValueError"/>, an ArgumentException.
        /// </summary>
        [TestMethod]
        public void OrderConverter_IsNumPysAcrossTheOrderTakingApis()
        {
            Action zerosQ = () => np.zeros(new Shape(2, 3), 'Q');
            zerosQ.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
            Action zerosA = () => np.zeros(new Shape(2, 3), 'a');
            zerosA.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            Action eyeK = () => np.eye(3, order: 'K');
            eyeK.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            Action copyQ = () => np.copy(np.arange(6).reshape(2, 3), 'Q');
            copyQ.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
            Action asarrayX = () => np.asarray(np.arange(6), order: 'X');
            asarrayX.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'X')");
            Action evaluateQ = () => np.evaluate((NumSharp.Backends.Iteration.NDExpr)np.arange(4.0) + 1.0, order: 'Q');
            evaluateQ.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
            np.copy(np.arange(6).reshape(2, 3), 'f').flags.f_contiguous.Should().BeTrue();

            // np.nditer: 'f' is F order — a C (2, 3) array is visited down its columns.
            var a = np.arange(6).reshape(2, 3);
            var seen = new List<long>();
            using (var it = np.nditer(a, order: 'f'))
                foreach (var v in it)
                    seen.Add(v[0].GetInt64());
            seen.Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
        }
    }
}
