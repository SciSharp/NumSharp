using System;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — fancy (advanced) indexing: the index's logical order and the result's layout
    // =====================================================================
    //
    // Found 2026-10-05 while giving Generator.permutation(x, axis > 0) NumPy's fancy-index layout (0f904216), then probed
    // against NumPy 2.4.2 over 62 getter / setter / empty cases. NumPy's mapping.c builds a fancy-index result in one of
    // three ways:
    //   (a) a 1-D source with ONE integer index array (the trivial path): the result takes the index array's order
    //       ("Same order as indices"), so an F-ordered or transposed index gives an F-ordered result;
    //   (b) every axis indexed by an array (no subspace): NpyIter allocates the result in the index arrays' memory order
    //       (NPY_KEEPORDER; C wins a tie);
    //   (c) otherwise MapIter fills a buffer holding the fancy dims (C order, or the index arrays' order when the rest has
    //       one element) followed by the remaining axes in the SOURCE's memory order (PyArray_CreateSortedStridePerm).
    //       When the fancy dims are not first (a slice or an ellipsis before them, mit->consec), it returns that buffer
    //       transposed back into place (PyArray_MapIterSwapAxes): a VIEW, owndata False.
    // NumSharp returns a fresh C-ordered array that owns its data on every route. The values agree, except in group 1.
    //
    // Group 1 (VALUES, "fancy-index-memory-order"): an N-D index array that is not C-contiguous (F-ordered or transposed)
    //   is read in MEMORY order when another index stands beside it — a slice, an ellipsis or an integer. The getter
    //   returns the wrong elements and the setter writes to the wrong positions, silently, with the right shape. A lone
    //   index array, arrays on every axis, and 1-D reversed / strided index arrays are read in logical order (the control).
    // Group 2 (layout, (c) with consec): a slice or an ellipsis before the index — NumPy's transposed view.
    // Group 3 (layout, (a) and (b)): a lone index or arrays on every axis — the result follows the index arrays' order.
    // Group 4 (layout, (c)): the remaining axes keep the SOURCE's memory order — an F-ordered or permuted 3-D source.
    //   Generator.permutation(x, axis > 0) rides route (c) through np.FancyIndexAlongAxis, which copies a non-C source to
    //   C order first, so its result has NumPy's view but not NumPy's strides for a non-C source of three or more dims.
    //
    // Every expected value, shape, stride and flag below is NumPy 2.4.2's answer for the same call (int64, byte strides).
    // EXPECTED (when fixed): each test passes; remove its [OpenBugs] and move it next to the indexing tests.
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>NumPy's <c>np.arange(12).reshape(4, 3)</c>.</summary>
        /// <returns>A fresh C-ordered int64 array.</returns>
        private static NDArray FancyA2() => np.arange(12).reshape(4, 3);

        /// <summary>NumPy's <c>np.arange(24).reshape(2, 3, 4)</c>.</summary>
        /// <returns>A fresh C-ordered int64 array.</returns>
        private static NDArray FancyA3() => np.arange(24).reshape(2, 3, 4);

        /// <summary>
        ///     The index <c>[[2, 0], [1, 1]]</c> laid out F-ordered (<c>np.asfortranarray</c>): its memory reads
        ///     <c>[2, 1, 0, 1]</c>, so a reader that walks memory instead of logical order sees <c>[[2, 1], [0, 1]]</c>.
        /// </summary>
        /// <returns>A fresh F-ordered int64 array.</returns>
        private static NDArray FancyIndexF() => np.asfortranarray(np.array(new long[,] { { 2, 0 }, { 1, 1 } }));

        /// <summary>
        ///     The index <c>[[2, 1], [0, 1]]</c> as a transposed view of <c>[[2, 0], [1, 1]]</c>: logical
        ///     <c>[[2, 1], [0, 1]]</c>, memory <c>[2, 0, 1, 1]</c>.
        /// </summary>
        /// <returns>A transposed (F-contiguous) view.</returns>
        private static NDArray FancyIndexT() => np.array(new long[,] { { 2, 0 }, { 1, 1 } }).T;

        // ----- Group 1: a non-C-contiguous N-D index is read in memory order beside another index (VALUES) -----

        /// <summary>
        ///     <c>a[:, idx]</c> with an F-ordered 2-D index: NumPy reads <c>idx</c> in logical order,
        ///     <c>np.arange(12).reshape(4, 3)[:, np.asfortranarray([[2, 0], [1, 1]])]</c> is <c>[[[2, 0], [1, 1]], [[5, 3],
        ///     [4, 4]], …]</c>. NumSharp walks the index's memory, <c>[2, 1, 0, 1]</c>, and returns <c>[[[2, 1], [0, 1]],
        ///     …]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-memory-order")]
        public void FancyGet_SliceThenFOrderedIndex_ShouldReadTheIndexInLogicalOrder()
        {
            using var a = FancyA2();
            using var idx = FancyIndexF();
            using var r = a[Slice.All, idx];

            r.shape.Should().Equal(new long[] { 4, 2, 2 });
            r.ToArray<long>().Should().Equal(new long[] { 2, 0, 1, 1, 5, 3, 4, 4, 8, 6, 7, 7, 11, 9, 10, 10 },
                "NumSharp reads the F-ordered index in memory order: [2, 1, 0, 1, 5, 4, 3, 4, …]");
        }

        /// <summary>
        ///     <c>a[:, idx.T]</c> with a transposed index: NumPy reads the logical <c>[[2, 1], [0, 1]]</c>; NumSharp walks the
        ///     memory <c>[2, 0, 1, 1]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-memory-order")]
        public void FancyGet_SliceThenTransposedIndex_ShouldReadTheIndexInLogicalOrder()
        {
            using var a = FancyA2();
            using var idx = FancyIndexT();
            using var r = a[Slice.All, idx];

            r.shape.Should().Equal(new long[] { 4, 2, 2 });
            r.ToArray<long>().Should().Equal(new long[] { 2, 1, 0, 1, 5, 4, 3, 4, 8, 7, 6, 7, 11, 10, 9, 10 },
                "NumSharp reads the transposed index in memory order: [2, 0, 1, 1, 5, 3, 4, 4, …]");
        }

        /// <summary>
        ///     <c>a[idx, :]</c>: the same misread with the index FIRST and an explicit slice after it (the lone <c>a[idx]</c>
        ///     is read correctly — see the control). NumPy: <c>[[[6, 7, 8], [0, 1, 2]], [[3, 4, 5], [3, 4, 5]]]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-memory-order")]
        public void FancyGet_FOrderedIndexThenSlice_ShouldReadTheIndexInLogicalOrder()
        {
            using var a = FancyA2();
            using var idx = FancyIndexF();
            using var r = a[idx, Slice.All];

            r.shape.Should().Equal(new long[] { 2, 2, 3 });
            r.ToArray<long>().Should().Equal(new long[] { 6, 7, 8, 0, 1, 2, 3, 4, 5, 3, 4, 5 },
                "NumSharp returns the rows in the index's memory order: [6, 7, 8, 3, 4, 5, 0, 1, 2, 3, 4, 5]");
        }

        /// <summary>
        ///     3-D: the misread with an ellipsis before the index (<c>a[..., idx]</c>, the last axis) and with a slice before
        ///     it on the middle axis (<c>a[:, idx]</c>). NumPy's values below; NumSharp swaps the two middle index entries.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-memory-order")]
        public void FancyGet_3D_FOrderedIndexBehindEllipsisOrSlice_ShouldReadTheIndexInLogicalOrder()
        {
            using var a = FancyA3();
            using var idx = FancyIndexF();

            using var last = a[Slice.Ellipsis, idx];
            last.shape.Should().Equal(new long[] { 2, 3, 2, 2 });
            last.ToArray<long>().Should().Equal(new long[]
            {
                2, 0, 1, 1, 6, 4, 5, 5, 10, 8, 9, 9, 14, 12, 13, 13, 18, 16, 17, 17, 22, 20, 21, 21,
            }, "a[..., idx] with an F-ordered idx");

            using var middle = a[Slice.All, idx];
            middle.shape.Should().Equal(new long[] { 2, 2, 2, 4 });
            middle.ToArray<long>().Should().Equal(new long[]
            {
                8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7,
                20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 18, 19,
            }, "a[:, idx] with an F-ordered idx");
        }

        /// <summary>
        ///     An integer beside the index: <c>a[1, idx]</c> (NumPy: rows <c>a[1][[2, 0], [1, 1]]</c>) and
        ///     <c>a[0, :, idx]</c> (non-consecutive, so NumPy puts the index dims first: shape <c>(2, 2, 3)</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-memory-order")]
        public void FancyGet_IntegerBesideFOrderedIndex_ShouldReadTheIndexInLogicalOrder()
        {
            using var a = FancyA3();
            using var idx = FancyIndexF();

            using var r1 = a[1, idx];
            r1.shape.Should().Equal(new long[] { 2, 2, 4 });
            r1.ToArray<long>().Should().Equal(new long[] { 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 18, 19 },
                "a[1, idx] with an F-ordered idx");

            using var r2 = a[0, Slice.All, idx];
            r2.shape.Should().Equal(new long[] { 2, 2, 3 });
            r2.ToArray<long>().Should().Equal(new long[] { 2, 6, 10, 0, 4, 8, 1, 5, 9, 1, 5, 9 },
                "a[0, :, idx] with an F-ordered idx");
        }

        /// <summary>
        ///     The SETTER writes through the same misread: <c>t[:, idx] = 100 + arange(8).reshape(2, 2, 2)</c> with
        ///     <c>idx = [[3, 0], [1, 2]]</c> (distinct targets) F-ordered, transposed, or behind an ellipsis. NumPy:
        ///     <c>t == [[101, 102, 103, 100], [105, 106, 107, 104]]</c>; NumSharp swaps the targets of the two middle values.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-memory-order")]
        public void FancySet_NonCIndexBesideASlice_ShouldWriteTheLogicalPositions()
        {
            var expected = new long[] { 101, 102, 103, 100, 105, 106, 107, 104 };
            using var idxF = np.asfortranarray(np.array(new long[,] { { 3, 0 }, { 1, 2 } }));
            using var idxT = np.array(new long[,] { { 3, 1 }, { 0, 2 } }).T;   // logical [[3, 0], [1, 2]]

            foreach (var (name, index) in new (string, object[])[]
                     {
                         ("t[:, idxF]", new object[] { Slice.All, idxF }),
                         ("t[:, idxT]", new object[] { Slice.All, idxT }),
                         ("t[..., idxF]", new object[] { Slice.Ellipsis, idxF }),
                     })
            {
                using var t = np.arange(8).reshape(2, 4).copy();
                using var values = (np.arange(8) + 100).reshape(2, 2, 2);
                t[index] = values;
                t.ToArray<long>().Should().Equal(expected,
                    $"{name} = v writes v[r, i, j] to t[r, idx[i, j]] (NumSharp: [102, 101, 103, 100, …])");
            }
        }

        /// <summary>
        ///     Control, passing today: the routes that read a non-C-contiguous index in logical order — a lone index array
        ///     (<c>a[idxF]</c>, <c>a[idx.T]</c>, a 1-D source), arrays on every axis (<c>a[idxF, idxC]</c>), and 1-D
        ///     reversed or stepped index views behind a slice, for the getter and the setter. A fix for group 1 must keep
        ///     these.
        /// </summary>
        [TestMethod]
        public void FancyIndex_LogicalOrderRoutes_ReadTheIndexCorrectly()
        {
            using var a1 = np.arange(10);
            using var a2 = FancyA2();
            using var idxF = FancyIndexF();
            using var idxT = FancyIndexT();
            using var idxC = np.array(new long[,] { { 2, 0 }, { 1, 1 } });
            using var rev = np.array(new long[] { 0, 1, 2 })["::-1"];
            using var step = np.array(new long[] { 2, 9, 0, 9, 1, 9 })["::2"];

            using (var r = a2[idxF]) r.ToArray<long>().Should().Equal(new long[] { 6, 7, 8, 0, 1, 2, 3, 4, 5, 3, 4, 5 });
            using (var r = a2[idxT]) r.ToArray<long>().Should().Equal(new long[] { 6, 7, 8, 3, 4, 5, 0, 1, 2, 3, 4, 5 });
            using (var r = a1[idxF]) r.ToArray<long>().Should().Equal(new long[] { 2, 0, 1, 1 });
            using (var r = a2[idxF, idxC]) r.ToArray<long>().Should().Equal(new long[] { 8, 0, 4, 4 });
            using (var r = a2[Slice.All, rev]) r.ToArray<long>().Should().Equal(new long[] { 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9 });
            using (var r = a2[Slice.All, step]) r.ToArray<long>().Should().Equal(new long[] { 2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10 });

            using var t = np.arange(8).reshape(2, 4).copy();
            using var d4r = np.array(new long[] { 2, 0, 3, 1 })["::-1"];   // logical [1, 3, 0, 2]
            using var values = (np.arange(8) + 100).reshape(2, 4);
            t[Slice.All, d4r] = values;
            t.ToArray<long>().Should().Equal(new long[] { 102, 100, 103, 101, 106, 104, 107, 105 });
        }

        // ----- Group 2: a slice or an ellipsis before the index — NumPy returns the buffer transposed back (a view) -----

        /// <summary>
        ///     <c>a[:, idx]</c> on a 2-D array is NumPy's transposed view: the buffer <c>(3, 4)</c> is C-ordered, so the
        ///     result <c>(4, 3)</c> has strides <c>(8, 32)</c>, is F-contiguous and does not own its data. The same for the
        ///     ellipsis spelling, an int32 index and a boolean mask (<c>a[:, [True, False, True]]</c>). NumSharp returns a
        ///     fresh C array (strides <c>(24, 8)</c>, owndata True).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-layout")]
        public void FancyGet_SliceThenIndex_ShouldBeNumPysTransposedView()
        {
            using var a = FancyA2();
            using var idx = np.array(new long[] { 2, 0, 1 });
            using var idx32 = np.array(new[] { 2, 0, 1 });
            using var mask = np.array(new[] { true, false, true });
            var values = new long[] { 2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10 };

            foreach (var (name, r, shape, expected) in new (string, NDArray, long[], long[])[]
                     {
                         ("a[:, idx]", a[Slice.All, idx], new long[] { 4, 3 }, values),
                         ("a[..., idx]", a[Slice.Ellipsis, idx], new long[] { 4, 3 }, values),
                         ("a[:, idx32]", a[Slice.All, idx32], new long[] { 4, 3 }, values),
                         ("a[:, mask]", a[Slice.All, mask], new long[] { 4, 2 }, new long[] { 0, 2, 3, 5, 6, 8, 9, 11 }),
                     })
            {
                using (r)
                {
                    r.shape.Should().Equal(shape, name);
                    r.ToArray<long>().Should().Equal(expected, name);
                    r.strides.Should().Equal(new long[] { 8, 32 }, $"{name}: NumPy's (rows-fancy) buffer transposed back");
                    r.flags.owndata.Should().BeFalse($"{name}: NumPy returns PyArray_MapIterSwapAxes' transpose, a view");
                    r.flags.f_contiguous.Should().BeTrue(name);
                    r.flags.c_contiguous.Should().BeFalse(name);
                }
            }
        }

        /// <summary>
        ///     3-D: the buffer holds the index dims first (C order) and the remaining axes after them, then is transposed
        ///     back — <c>a[:, idx]</c> and <c>a[:, idx, :]</c> have strides <c>(32, 64, 8)</c>, <c>a[:, :, idx]</c> and
        ///     <c>a[..., idx]</c> have <c>(24, 8, 48)</c>; all views. NumSharp: C strides <c>(96, 32, 8)</c> / <c>(72, 24, 8)</c>,
        ///     owning.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-layout")]
        public void FancyGet_3D_SliceThenIndex_ShouldHaveNumPysPermutedStrides()
        {
            using var a = FancyA3();
            using var idx = np.array(new long[] { 2, 0, 1 });
            var middle = new long[] { 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19 };
            var last = new long[] { 2, 0, 1, 6, 4, 5, 10, 8, 9, 14, 12, 13, 18, 16, 17, 22, 20, 21 };

            foreach (var (name, r, strides, expected) in new (string, NDArray, long[], long[])[]
                     {
                         ("a[:, idx]", a[Slice.All, idx], new long[] { 32, 64, 8 }, middle),
                         ("a[:, idx, :]", a[Slice.All, idx, Slice.All], new long[] { 32, 64, 8 }, middle),
                         ("a[:, :, idx]", a[Slice.All, Slice.All, idx], new long[] { 24, 8, 48 }, last),
                         ("a[..., idx]", a[Slice.Ellipsis, idx], new long[] { 24, 8, 48 }, last),
                     })
            {
                using (r)
                {
                    r.ToArray<long>().Should().Equal(expected, name);
                    r.strides.Should().Equal(strides, $"{name}: NumPy's buffer (index dims first) transposed back");
                    r.flags.owndata.Should().BeFalse($"{name}: a view of the MapIter buffer");
                }
            }
        }

        /// <summary>
        ///     A 2-D index behind a slice, two index arrays behind a slice, and the one-element subspace (NumPy takes the
        ///     index's own order for the fancy dims there): <c>a[:, [[2, 0], [1, 1]]]</c> strides <c>(8, 64, 32)</c>;
        ///     <c>b[:, [2, 0], [4, 1]]</c> on <c>arange(60).reshape(3, 4, 5)</c> strides <c>(8, 24)</c> (F); <c>c[:, idx]</c>
        ///     on <c>arange(3).reshape(1, 3)</c> strides <c>(8, 16, 8)</c> (C). All views; NumSharp owns C arrays.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-layout")]
        public void FancyGet_SliceThenMultiDimOrMultipleIndices_ShouldHaveNumPysPermutedStrides()
        {
            using var a = FancyA2();
            using var b = np.arange(60).reshape(3, 4, 5);
            using var c = np.arange(3).reshape(1, 3);
            using var idx2 = np.array(new long[,] { { 2, 0 }, { 1, 1 } });
            using var rows = np.array(new long[] { 2, 0 });
            using var cols = np.array(new long[] { 4, 1 });

            foreach (var (name, r, strides, expected) in new (string, NDArray, long[], long[])[]
                     {
                         ("a[:, idx2]", a[Slice.All, idx2], new long[] { 8, 64, 32 },
                             new long[] { 2, 0, 1, 1, 5, 3, 4, 4, 8, 6, 7, 7, 11, 9, 10, 10 }),
                         ("b[:, rows, cols]", b[Slice.All, rows, cols], new long[] { 8, 24 }, new long[] { 14, 1, 34, 21, 54, 41 }),
                         ("c[:, idx2]", c[Slice.All, idx2], new long[] { 8, 16, 8 }, new long[] { 2, 0, 1, 1 }),
                     })
            {
                using (r)
                {
                    r.ToArray<long>().Should().Equal(expected, name);
                    r.strides.Should().Equal(strides, $"{name}: NumPy's MapIter buffer transposed back");
                    r.flags.owndata.Should().BeFalse($"{name}: a view of the MapIter buffer");
                }
            }
        }

        /// <summary>
        ///     An EMPTY index behind a slice is still NumPy's transposed view: <c>a[:, np.array([], int64)]</c> is
        ///     <c>(4, 0)</c> with owndata False. (Its strides are the empty-array rule's, pinned in
        ///     <c>OpenBugs.EmptyStrides.cs</c>, so they are not compared here.)
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-layout")]
        public void FancyGet_SliceThenEmptyIndex_ShouldBeAView()
        {
            using var a = FancyA2();
            using var empty = np.array(new long[0]);
            using var r = a[Slice.All, empty];

            r.shape.Should().Equal(new long[] { 4, 0 });
            r.flags.owndata.Should().BeFalse("NumPy transposes its (0, 4) buffer back into place, a view");
        }

        // ----- Group 3: a lone index or arrays on every axis — the result follows the index arrays' order -----

        /// <summary>
        ///     Routes (a) and (b): a 1-D source with an F-ordered, int32 F-ordered or transposed 2-D index, and arrays on
        ///     every axis of a 2-D source (both F, or a transposed 2-D index broadcast against a 1-D one), give NumPy an
        ///     F-ordered <c>(2, 2)</c> result, strides <c>(8, 16)</c>, that owns its data. NumSharp: C, <c>(16, 8)</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-layout")]
        public void FancyGet_IndexOnlyRoutes_ShouldFollowTheIndexArrayOrder()
        {
            using var a1 = np.arange(10);
            using var a2 = FancyA2();
            using var idxF = FancyIndexF();
            using var idxF32 = idxF.astype(np.int32);
            using var idxT = FancyIndexT();
            using var rev2 = np.array(new long[] { 0, 1, 2 })["::-1"][":2"];   // [2, 1]

            foreach (var (name, r, expected) in new (string, NDArray, long[])[]
                     {
                         ("a1[idxF]", a1[idxF], new long[] { 2, 0, 1, 1 }),
                         ("a1[idxF int32]", a1[idxF32], new long[] { 2, 0, 1, 1 }),
                         ("a1[idxT]", a1[idxT], new long[] { 2, 1, 0, 1 }),
                         ("a2[idxF, idxF]", a2[idxF, idxF], new long[] { 8, 0, 4, 4 }),
                         ("a2[idxT, rev2]", a2[idxT, rev2], new long[] { 8, 4, 2, 4 }),
                     })
            {
                using (r)
                {
                    r.ToArray<long>().Should().Equal(expected, name);
                    r.strides.Should().Equal(new long[] { 8, 16 }, $"{name}: NumPy allocates in the index arrays' order");
                    r.flags.f_contiguous.Should().BeTrue(name);
                    r.flags.owndata.Should().BeTrue(name);
                }
            }
        }

        // ----- Group 4: the remaining axes keep the source's memory order -----

        /// <summary>
        ///     Route (c) lays the remaining axes out in the SOURCE's memory order. With <c>f = asfortranarray(arange(24)
        ///     .reshape(2, 3, 4))</c>: <c>f[[1, 0]]</c> has strides <c>(96, 8, 24)</c> and owns its data (index first, no
        ///     transpose); <c>f[:, idx]</c> <c>(8, 64, 16)</c> and <c>f[:, :, idx]</c> <c>(8, 16, 48)</c> (F) are views. With
        ///     <c>p = arange(24).reshape(2, 3, 4).transpose(2, 0, 1)</c>: <c>p[:, [1, 0]]</c> <c>(8, 96, 32)</c>. NumSharp lays
        ///     every result out C-ordered.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-layout")]
        public void FancyGet_NonCSource_ShouldKeepTheSourceOrderOfTheOtherAxes()
        {
            using var f = np.asfortranarray(FancyA3());
            using var p = np.transpose(FancyA3(), new[] { 2, 0, 1 });
            using var idx = np.array(new long[] { 2, 0, 1 });
            using var pair = np.array(new long[] { 1, 0 });

            using (var r = f[pair])
            {
                r.ToArray<long>().Should().Equal(new long[] { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });
                r.strides.Should().Equal(new long[] { 96, 8, 24 }, "f[[1, 0]]: the (3, 4) rest keeps f's F order");
                r.flags.owndata.Should().BeTrue("the index is first, so NumPy returns its buffer itself");
            }

            using (var r = f[Slice.All, idx])
            {
                r.ToArray<long>().Should().Equal(new long[] { 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19 });
                r.strides.Should().Equal(new long[] { 8, 64, 16 }, "f[:, idx]");
                r.flags.owndata.Should().BeFalse("f[:, idx]");
            }

            using (var r = f[Slice.All, Slice.All, idx])
            {
                r.ToArray<long>().Should().Equal(new long[] { 2, 0, 1, 6, 4, 5, 10, 8, 9, 14, 12, 13, 18, 16, 17, 22, 20, 21 });
                r.strides.Should().Equal(new long[] { 8, 16, 48 }, "f[:, :, idx]");
                r.flags.f_contiguous.Should().BeTrue("f[:, :, idx]");
            }

            using (var r = p[Slice.All, pair])
            {
                r.ToArray<long>().Should().Equal(new long[] { 12, 16, 20, 0, 4, 8, 13, 17, 21, 1, 5, 9, 14, 18, 22, 2, 6, 10, 15, 19, 23, 3, 7, 11 });
                r.strides.Should().Equal(new long[] { 8, 96, 32 }, "p[:, [1, 0]]");
            }
        }

        /// <summary>
        ///     <c>Generator.permutation(x, axis &gt; 0)</c> is NumPy's <c>x[tuple(slices)]</c>, so for a non-C source of three
        ///     or more dims it takes route (c)'s source order too. NumPy, <c>default_rng(42)</c>: <c>permutation(f, 1)</c> has
        ///     strides <c>(8, 64, 16)</c>, <c>permutation(f, 2)</c> <c>(8, 16, 48)</c> (F), <c>permutation(p, 2)</c>
        ///     <c>(8, 32, 64)</c>. NumSharp's <c>np.FancyIndexAlongAxis</c> copies a non-C source to C first: <c>(32, 64, 8)</c>,
        ///     <c>(24, 8, 48)</c>, <c>(16, 8, 64)</c> — right values, NumPy's view, wrong strides (a gap of <c>0f904216</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "fancy-index-layout")]
        public void Permutation_NonLeadingAxisOfNonCSource_ShouldKeepTheSourceOrder()
        {
            using var f = np.asfortranarray(FancyA3());
            using var p = np.transpose(FancyA3(), new[] { 2, 0, 1 });

            using (var r = np.random.default_rng(42).permutation(f, 1))
            {
                r.ToArray<long>().Should().Equal(new long[] { 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15 });
                r.flags.owndata.Should().BeFalse();
                r.strides.Should().Equal(new long[] { 8, 64, 16 }, "permutation(f, 1)");
            }

            using (var r = np.random.default_rng(42).permutation(f, 2))
            {
                r.ToArray<long>().Should().Equal(new long[] { 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20 });
                r.strides.Should().Equal(new long[] { 8, 16, 48 }, "permutation(f, 2)");
                r.flags.f_contiguous.Should().BeTrue("permutation(f, 2)");
            }

            using (var r = np.random.default_rng(42).permutation(p, 2))
            {
                r.ToArray<long>().Should().Equal(new long[] { 8, 4, 0, 20, 16, 12, 9, 5, 1, 21, 17, 13, 10, 6, 2, 22, 18, 14, 11, 7, 3, 23, 19, 15 });
                r.strides.Should().Equal(new long[] { 8, 32, 64 }, "permutation(p, 2)");
            }
        }
    }
}
