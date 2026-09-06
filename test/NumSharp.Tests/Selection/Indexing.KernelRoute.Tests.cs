using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    /// Pins the kernel route behind the single-index-array fancy get/set (<c>a[idx]</c>,
    /// <c>a[idx] = v</c>, <c>m[ridx]</c>, <c>m[ridx] = row</c>) — the take/put IL kernels reading the
    /// index buffer in place at its own width (int32 or int64) — and the SIMD run scan of the
    /// ufunc <c>where=</c> masked driver. Every expected value is what NumPy 2.4.2 produces for the
    /// same expression; the semantics that are easy to lose when a route changes are the ones
    /// pinned explicitly: negative-index normalisation, the verbatim IndexError, NO partial write
    /// on a bad index (NumPy validates every index before its first store), view offsets on both
    /// the source and the index array, the value-broadcast rules and their two ValueError texts,
    /// last-write-wins on duplicate indices, and every dtype width (1/2/4/8/16 bytes).
    /// </summary>
    [TestClass]
    public class Indexing_KernelRoute_Tests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        private static long[] Longs(NDArray a) => a.flat.ToArray<long>();

        private static double[] Doubles(NDArray a)
        {
            var f = a.astype(NPTypeCode.Double).flat;
            var r = new double[f.size];
            for (long i = 0; i < f.size; i++) r[i] = f.GetAtIndex<double>(i);
            return r;
        }

        // ───────────────────────────── GET ─────────────────────────────

        [TestMethod]
        public void Get1D_Int32AndInt64Indices_MatchElementwiseReference()
        {
            var a = np.arange(1000, dtype: np.float64) * 0.5;
            var idxL = new long[300];
            for (int i = 0; i < idxL.Length; i++)
                idxL[i] = ((i * 7919L) % 1000) * (i % 3 == 0 ? -1 : 1);     // mixed sign, wraps, repeats
            var expected = idxL.Select(k => (k < 0 ? k + 1000 : k) * 0.5).ToArray();

            var r64 = a[np.array(idxL)];
            var r32 = a[np.array(idxL.Select(x => (int)x).ToArray())];

            r64.shape.Should().Equal(new long[] { 300 });
            Doubles(r64).Should().Equal(expected);
            Doubles(r32).Should().Equal(expected);
            r64.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Get1D_EveryDtype_Width1To16()
        {
            var idx = np.array(new[] { 3, -1, 0, 49, 7, 7 });
            foreach (var tc in AllDtypes)
            {
                var a = np.arange(50).astype(tc);
                var r = a[idx];
                r.typecode.Should().Be(tc, tc.ToString());
                r.shape.Should().Equal(new long[] { 6 });
                var expected = tc == NPTypeCode.Boolean ? new double[] { 1, 1, 0, 1, 1, 1 } : new double[] { 3, 49, 0, 49, 7, 7 };
                Doubles(r).Should().Equal(expected, tc.ToString());
            }
        }

        [TestMethod]
        public void Get1D_OutOfBounds_VerbatimIndexError()
        {
            var a = np.arange(1000, dtype: np.float64);
            Action high = () => { var _ = a[np.array(new long[] { 0, 1000, 2 })]; };
            high.Should().Throw<IndexError>().WithMessage("index 1000 is out of bounds for axis 0 with size 1000");
            Action low = () => { var _ = a[np.array(new[] { 5, -1001 })]; };
            low.Should().Throw<IndexError>().WithMessage("index -1001 is out of bounds for axis 0 with size 1000");
        }

        [TestMethod]
        public void Get1D_SourceIsAnOffsetContiguousView()
        {
            var a = np.arange(1000, dtype: np.int64);
            var v = a["5:"];                                   // C-contiguous, offset 5
            var r = v[np.array(new long[] { 0, 4, -1, 994 })];
            Longs(r).Should().Equal(5, 9, 999, 999);
            Action oob = () => { var _ = v[np.array(new long[] { 995 })]; };
            oob.Should().Throw<IndexError>().WithMessage("index 995 is out of bounds for axis 0 with size 995");
        }

        [TestMethod]
        public void Get1D_IndexArrayIsAnOffsetContiguousView()
        {
            var a = np.arange(100, dtype: np.int64) * 10;
            var big = np.array(new long[] { 99, 99, 1, 2, 3 });
            var idx = big["2:"];                                // int64 view, offset 2 → [1,2,3]
            Longs(a[idx]).Should().Equal(10, 20, 30);
            var big32 = np.array(new[] { 99, 99, 4, 5 });
            Longs(a[big32["2:"]]).Should().Equal(40, 50);
        }

        [TestMethod]
        public void Get1D_StridedIndexView_TakesTheGeneralRoute_SameValues()
        {
            var a = np.arange(100, dtype: np.int64) * 10;
            var idx = np.array(new long[] { 1, 90, 2, 90, 3, 90 })["::2"];   // non-contiguous index → general route
            Longs(a[idx]).Should().Equal(10, 20, 30);
            var rev = np.array(new long[] { 1, 2, 3 })["::-1"];
            Longs(a[rev]).Should().Equal(30, 20, 10);
        }

        [TestMethod]
        public void Get_NDIndexArray_ResultTakesTheIndexShape()
        {
            var a = np.arange(10, dtype: np.int64) * 10;
            var idx = np.array(new long[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var r = a[idx];
            r.shape.Should().Equal(new long[] { 2, 3 });
            Longs(r).Should().Equal(10, 20, 30, 40, 50, 60);
            var s = a[NDArray.Scalar(7L)];                            // 0-d index → 0-d result
            s.ndim.Should().Be(0);
            s.GetAtIndex<long>(0).Should().Be(70);
        }

        [TestMethod]
        public void Get_RowGather_2D_And_3D_MatchesTakeAxis0()
        {
            var m = np.arange(800, dtype: np.float64).reshape(100, 8);
            var ridx = np.array(new long[] { 5, -1, 0, 5, 99 });
            var r = m[ridx];
            r.shape.Should().Equal(new long[] { 5, 8 });
            Assert.IsTrue(np.array_equal(r, np.take(m, ridx, axis: 0)));
            Doubles(r["1"]).Should().Equal(Enumerable.Range(792, 8).Select(x => (double)x).ToArray());

            var c = np.arange(120, dtype: np.int32).reshape(10, 4, 3);
            var rc = c[np.array(new[] { 9, 0, -10 })];
            rc.shape.Should().Equal(new long[] { 3, 4, 3 });
            Assert.IsTrue(np.array_equal(rc, np.take(c, np.array(new[] { 9, 0, 0 }), axis: 0)));

            // a 2-D index over rows: (2,2) index → (2,2,8)
            var r2 = m[np.array(new long[] { 1, 2, 3, 4 }).reshape(2, 2)];
            r2.shape.Should().Equal(new long[] { 2, 2, 8 });
            Doubles(r2["1, 1"]).Should().Equal(Enumerable.Range(32, 8).Select(x => (double)x).ToArray());

            Action oob = () => { var _ = m[np.array(new long[] { 0, 100 })]; };
            oob.Should().Throw<IndexError>().WithMessage("index 100 is out of bounds for axis 0 with size 100");
        }

        [TestMethod]
        public void Get_RowGather_NarrowSlab_UsesTypedWidth()
        {
            // uint8 rows of 8 bytes → the 8-byte typed flat kernel; int16 rows of 2 → 4-byte width
            var m = np.arange(80).astype(np.uint8).reshape(10, 8);
            var r = m[np.array(new[] { 9, -10, 3 })];
            r.shape.Should().Equal(new long[] { 3, 8 });
            Doubles(r["0"]).Should().Equal(Enumerable.Range(72, 8).Select(x => (double)x).ToArray());
            Doubles(r["1"]).Should().Equal(Enumerable.Range(0, 8).Select(x => (double)x).ToArray());

            var s = np.arange(20).astype(np.int16).reshape(10, 2);
            Doubles(s[np.array(new long[] { 4, 4, 0 })]).Should().Equal(8, 9, 8, 9, 0, 1);
        }

        [TestMethod]
        public void Get_EmptyTrailingAxis_ShapeOnly()
        {
            var z = np.zeros(new Shape(3, 0), np.float64);
            var r = z[np.array(new long[] { 0, 1 })];
            r.shape.Should().Equal(new long[] { 2, 0 });
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void Get_ReturnsAFreshOwnedCopy()
        {
            var a = np.arange(10, dtype: np.int64);
            var r = a[np.array(new long[] { 1, 2 })];
            r[0] = 999;
            a.GetAtIndex<long>(1).Should().Be(1);
            r.flags.owndata.Should().BeTrue();
        }

        // ───────────────────────────── SET ─────────────────────────────

        [TestMethod]
        public void Set1D_ArrayScalarAndBroadcastValues()
        {
            var a = np.zeros(new Shape(10), np.float64);
            a[np.array(new long[] { 1, -1, 3 })] = np.array(new[] { 1.5, 2.5, 3.5 });
            Doubles(a).Should().Equal(0, 1.5, 0, 3.5, 0, 0, 0, 0, 0, 2.5);

            a[np.array(new[] { 0, 2 })] = 7.0;                                     // scalar
            Doubles(a).Should().Equal(7, 1.5, 7, 3.5, 0, 0, 0, 0, 0, 2.5);

            a[np.array(new[] { 4, 5, 6 })] = np.array(new[] { 9.0 });               // (1,) broadcast → materialised
            Doubles(a).Should().Equal(7, 1.5, 7, 3.5, 9, 9, 9, 0, 0, 2.5);

            a[np.array(new long[] { 7, 8 }).reshape(2, 1)] = np.array(new[] { 4.0, 5.0 }).reshape(2, 1);   // 2-D index, exact shape
            Doubles(a).Should().Equal(7, 1.5, 7, 3.5, 9, 9, 9, 4, 5, 2.5);
        }

        [TestMethod]
        public void Set1D_OutOfBounds_RaisesBeforeAnyWrite()
        {
            var a = np.zeros(new Shape(5), np.float64);
            Action act = () => a[np.array(new long[] { 0, 1, 10 })] = np.array(new[] { 1.0, 2.0, 3.0 });
            act.Should().Throw<IndexError>().WithMessage("index 10 is out of bounds for axis 0 with size 5");
            Doubles(a).Should().Equal(0, 0, 0, 0, 0);                              // NumPy: nothing written

            Action neg = () => a[np.array(new[] { 0, -6 })] = 1.0;
            neg.Should().Throw<IndexError>().WithMessage("index -6 is out of bounds for axis 0 with size 5");
            Doubles(a).Should().Equal(0, 0, 0, 0, 0);

            // the SIMD bounds scan: a bad index deep inside a long array is found and named
            var big = np.zeros(new Shape(1000), np.int64);
            var idx = np.arange(1000, dtype: np.int64);
            idx[777] = 1000;
            Action deep = () => big[idx] = 1L;
            deep.Should().Throw<IndexError>().WithMessage("index 1000 is out of bounds for axis 0 with size 1000");
            Longs(big).Sum().Should().Be(0);
        }

        [TestMethod]
        public void Set1D_IncompatibleValue_VerbatimValueError()
        {
            var a = np.zeros(new Shape(10), np.float64);
            Action act = () => a[np.array(new long[] { 1, 2, 3 })] = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            act.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (5,) into shape (3,)");
        }

        [TestMethod]
        public void Set1D_DuplicateIndices_LastWriteWins()
        {
            var a = np.zeros(new Shape(4), np.int64);
            a[np.array(new long[] { 1, 1, 1, 2 })] = np.array(new long[] { 5, 6, 7, 8 });
            Longs(a).Should().Equal(0, 7, 8, 0);
        }

        [TestMethod]
        public void Set1D_EveryDtype_Width1To16()
        {
            foreach (var tc in AllDtypes)
            {
                var a = np.zeros(new Shape(6), tc);
                var vals = np.arange(1, 4).astype(tc);
                a[np.array(new[] { 0, -1, 2 })] = vals;
                var expected = tc == NPTypeCode.Boolean ? new double[] { 1, 0, 1, 0, 0, 1 } : new double[] { 1, 0, 3, 0, 0, 2 };
                Doubles(a).Should().Equal(expected, tc.ToString());
                a[np.array(new long[] { 1 })] = np.arange(5, 6).astype(tc);
                Doubles(a)[1].Should().Be(tc == NPTypeCode.Boolean ? 1 : 5, tc.ToString());
            }
        }

        [TestMethod]
        public void Set1D_ValueDtypeCastsToDestination()
        {
            var a = np.zeros(new Shape(4), np.int64);
            a[np.array(new[] { 0, 1 })] = np.array(new[] { 1.9, -2.9 });
            Longs(a).Should().Equal(1, -2, 0, 0);
        }

        [TestMethod]
        public void Set1D_OffsetViews_OnDestinationAndIndex()
        {
            var a = np.zeros(new Shape(10), np.int64);
            var v = a["4:"];                                                          // offset 4, size 6
            var idxBig = np.array(new long[] { 99, 0, 5, -1 });
            v[idxBig["1:"]] = np.array(new long[] { 1, 2, 3 });
            Longs(a).Should().Equal(0, 0, 0, 0, 1, 0, 0, 0, 0, 3);                  // v[0]=1, v[5]=2 then v[-1]=v[5]=3
            Action oob = () => v[np.array(new long[] { 6 })] = 1L;
            oob.Should().Throw<IndexError>().WithMessage("index 6 is out of bounds for axis 0 with size 6");
        }

        [TestMethod]
        public void Set2D_RowScatter_ExactSingleRowAndScalar()
        {
            var m = np.zeros(new Shape(5, 4), np.float64);
            var ridx = np.array(new long[] { 4, 0 });

            m[ridx] = np.arange(8, dtype: np.float64).reshape(2, 4);                 // exact (2,4)
            Doubles(m["4"]).Should().Equal(0, 1, 2, 3);
            Doubles(m["0"]).Should().Equal(4, 5, 6, 7);

            m[np.array(new[] { 1, 2 })] = np.array(new[] { 9.0, 8.0, 7.0, 6.0 });      // one row → every selected row (no copy)
            Doubles(m["1"]).Should().Equal(9, 8, 7, 6);
            Doubles(m["2"]).Should().Equal(9, 8, 7, 6);

            m[np.array(new long[] { 3, -1 })] = -1.0;                                  // scalar → materialised broadcast
            Doubles(m["3"]).Should().Equal(-1, -1, -1, -1);
            Doubles(m["4"]).Should().Equal(-1, -1, -1, -1);

            m[np.array(new long[] { 0 })] = np.array(new[] { 5.0 }).reshape(1, 1);     // (1,1) broadcasts to (1,4)
            Doubles(m["0"]).Should().Equal(5, 5, 5, 5);
        }

        [TestMethod]
        public void Set2D_RowScatter_NoPartialWriteAndVerbatimErrors()
        {
            var m = np.zeros(new Shape(5, 4), np.float64);
            Action oob = () => m[np.array(new long[] { 0, 1, 5 })] = 1.0;
            oob.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 5");
            Doubles(m).Sum().Should().Be(0);

            Action bad = () => m[np.array(new long[] { 0, 1 })] = np.array(new[] { 1.0, 2.0, 3.0 });
            bad.Should().Throw<ValueError>().WithMessage("shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,4)");
        }

        [TestMethod]
        public void Set2D_RowScatter_3DDestination_16ByteDtype()
        {
            var m = np.zeros(new Shape(4, 2, 3), NPTypeCode.Complex);
            var row = np.arange(6).astype(NPTypeCode.Complex).reshape(2, 3);
            m[np.array(new[] { 3, 1 })] = row;
            Assert.IsTrue(np.array_equal(m["3"], row));
            Assert.IsTrue(np.array_equal(m["1"], row));
            m.GetAtIndex<Complex>(0).Should().Be(Complex.Zero);
        }

        // ───────────────────────────── np.take / np.put int32 in place ─────────────────────────────

        [TestMethod]
        public void Take_Int32Indices_ReadInPlace_AllModes()
        {
            var a = np.arange(100, dtype: np.float64);
            var i32 = np.array(new[] { 5, -1, 150, -120 });
            var i64 = i32.astype(np.int64);
            Assert.IsTrue(np.array_equal(np.take(a, i32, mode: "wrap"), np.take(a, i64, mode: "wrap")));
            Assert.IsTrue(np.array_equal(np.take(a, i32, mode: "clip"), np.take(a, i64, mode: "clip")));
            Doubles(np.take(a, np.array(new[] { 5, -1 }))).Should().Equal(5, 99);
            Action raise = () => np.take(a, i32);
            raise.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*index 150 is out of bounds for axis with size 100*");

            var m = np.arange(24, dtype: np.int64).reshape(6, 4);
            Assert.IsTrue(np.array_equal(np.take(m, np.array(new[] { 5, 0 }), axis: 0), np.take(m, np.array(new long[] { 5, 0 }), axis: 0)));
            Assert.IsTrue(np.array_equal(np.take(m, np.array(new[] { 3, -1 }), axis: 1), np.take(m, np.array(new long[] { 3, -1 }), axis: 1)));
        }

        [TestMethod]
        public void Put_Int32Indices_ReadInPlace_AllModes()
        {
            var a = np.zeros(new Shape(10), np.int64);
            np.put(a, np.array(new[] { 1, -1 }), np.array(new long[] { 5, 6 }));
            Longs(a).Should().Equal(0, 5, 0, 0, 0, 0, 0, 0, 0, 6);
            np.put(a, np.array(new[] { 12, -13 }), np.array(new long[] { 7 }), mode: "wrap");
            Longs(a).Should().Equal(0, 5, 7, 0, 0, 0, 0, 7, 0, 6);
            np.put(a, np.array(new[] { 99 }), np.array(new long[] { 8 }), mode: "clip");
            Longs(a)[9].Should().Be(8);
            Action raise = () => np.put(a, np.array(new[] { 0, 10 }), np.array(new long[] { 1 }));
            raise.Should().Throw<IndexOutOfRangeException>().WithMessage("index 10 is out of bounds for axis 0 with size 10");
        }

        [TestMethod]
        public void Put_ValuesCycle_ThroughTheFlatKernel()
        {
            var a = np.zeros(new Shape(7), np.int32);
            np.put(a, np.arange(7), np.array(new[] { 1, 2, 3 }));
            Doubles(a).Should().Equal(1, 2, 3, 1, 2, 3, 1);
        }

        [TestMethod]
        public void FlatKernels_AreAvailableForEveryTypedWidth()
        {
            foreach (int width in new[] { 1, 2, 4, 8, 16 })
            {
                DirectILKernelGenerator.GetTakeFlatKernel(width, idx32: true, prefetch: false).Should().NotBeNull();
                DirectILKernelGenerator.GetTakeFlatKernel(width, idx32: false, prefetch: true).Should().NotBeNull();
                DirectILKernelGenerator.GetPutFlatKernel(width, idx32: true).Should().NotBeNull();
                DirectILKernelGenerator.GetPutFlatKernel(width, idx32: false).Should().NotBeNull();
            }
            DirectILKernelGenerator.GetTakeFlatKernel(0, false, false).Should().BeNull();       // runtime-sized slabs stay on the general kernel
        }

        // ───────────────────────────── where= run scan ─────────────────────────────

        [TestMethod]
        public void Where_ContiguousMask_RunBoundariesAcrossVectorBlocks()
        {
            const int n = 1000;
            var a = np.arange(n, dtype: np.float64);
            var b = np.arange(n, dtype: np.float64) * 2;
            var ar = np.arange(n);
            var masks = new (string name, NDArray mask)[]
            {
                ("alternating", (ar % 2) == 0),
                ("runs3", (np.floor_divide(ar, 3) % 2) == 0),
                ("runs31", (np.floor_divide(ar, 31) % 2) == 0),
                ("runs32", (np.floor_divide(ar, 32) % 2) == 0),
                ("runs33", (np.floor_divide(ar, 33) % 2) == 0),
                ("runs100", (np.floor_divide(ar, 100) % 2) == 0),
                ("allTrue", ar >= 0),
                ("allFalse", ar < 0),
                ("lastOnly", ar == n - 1),
                ("firstOnly", ar == 0),
                ("tail", ar >= n - 5),
                ("head", ar < 40),
                ("single700", ar == 700),
            };
            foreach (var (name, mask) in masks)
            {
                var o = np.full(new Shape(n), -1.0, np.float64);
                np.add(a, b, o, where: mask);
                var m = mask.ToArray<bool>();
                var expected = Enumerable.Range(0, n).Select(i => m[i] ? 3.0 * i : -1.0).ToArray();
                Doubles(o).Should().Equal(expected, name);
            }
        }

        // ───────────────────────────── where= on narrow strided rows: the masked 2-D block kernel ─────────────────────────────

        private static void AssertMaskedRowsMatchReference(NPTypeCode tc, int w, string maskKind)
        {
            const int rows = 257;                                                   // odd, well past one vector
            var back = ((np.arange(rows * 2 * w) * 7 + 1) % 13 + 1).astype(tc).reshape(rows, 2 * w);
            var back2 = ((np.arange(rows * 2 * w) * 3 + 5) % 13 + 1).astype(tc).reshape(rows, 2 * w);
            var sv = back[$":, :{w}"];                                              // strided rows, contiguous inner
            var sv2 = back2[$":, {w}:"];                                             // offset column view
            var ar = np.arange(rows * w).reshape(rows, w);
            NDArray mask = maskKind switch
            {
                "bytes" => (np.floor_divide(ar, 3) % 2) == 0,                       // a mask byte per element
                "rowmask" => ((np.arange(rows) % 2) == 0)[":, np.newaxis"],           // (rows,1): one byte per row
                "colmask" => ((np.arange(w) % 2) == 0)["np.newaxis, :"],              // (1,w): a row broadcast down
                _ => throw new ArgumentException(maskKind),
            };
            var maskFull = np.broadcast_to(mask, new Shape(rows, w));
            var prior = (np.arange(rows * w) % 5).astype(tc).reshape(rows, w);

            var o = prior.copy();
            np.add(sv, sv2, o, where: mask);
            var expected = np.where(maskFull, np.add(sv.copy(), sv2.copy()), prior);   // independent composition
            Assert.IsTrue(np.array_equal(o, expected), $"{tc} w={w} {maskKind} add");

            var o2 = prior.copy();
            np.multiply(sv, NDArray.Scalar(2).astype(tc), o2, where: mask);           // CS pattern (scalar rhs)
            var expected2 = np.where(maskFull, np.multiply(sv.copy(), NDArray.Scalar(2).astype(tc)), prior);
            Assert.IsTrue(np.array_equal(o2, expected2), $"{tc} w={w} {maskKind} multiply-scalar");

            if (tc == NPTypeCode.Double || tc == NPTypeCode.Single)
            {
                var o3 = prior.copy();
                np.sqrt(sv, o3, where: mask);                                          // unary, vector body
                Assert.IsTrue(np.array_equal(o3, np.where(maskFull, np.sqrt(sv.copy()), prior)), $"{tc} w={w} {maskKind} sqrt");
                var o4 = prior.copy();
                np.exp(sv, o4, where: mask);                                           // unary, scalar-only block
                Assert.IsTrue(np.array_equal(o4, np.where(maskFull, np.exp(sv.copy()), prior)), $"{tc} w={w} {maskKind} exp");
            }
        }

        [TestMethod]
        public void Where_NarrowStridedRows_EveryMaskKind_MatchesReference()
        {
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.UInt32,
                                       NPTypeCode.Int16, NPTypeCode.Byte, NPTypeCode.Half, NPTypeCode.Complex })
                foreach (int w in new[] { 1, 3, 4, 5, 8, 9, 17 })                      // sub-vector, one-vector, tail rows
                    foreach (var kind in new[] { "bytes", "rowmask", "colmask" })
                        AssertMaskedRowsMatchReference(tc, w, kind);
        }

        [TestMethod]
        public void Where_NarrowRows_3D_LeadingAxisWalkedPerBlock()
        {
            const int p = 5, q = 33, w = 4;
            var x = ((np.arange(p * q * 2 * w) * 3 + 1) % 11 + 1).astype(np.float64).reshape(p, q, 2 * w);
            var y = ((np.arange(p * q * 2 * w) * 5 + 2) % 11 + 1).astype(np.float64).reshape(p, q, 2 * w);
            foreach (var (sv, sv2) in new[] { (x[":, :, :4"], y[":, :, :4"]), (x["::2, :, :4"], y["::2, :, :4"]) })
            {
                var shape = sv.Shape;
                var n = (int)sv.size;
                var mask = ((np.arange(n) * 7) % 3 == 0).reshape(shape);
                var prior = (np.arange(n) % 5).astype(np.float64).reshape(shape);
                var o = prior.copy();
                np.add(sv, sv2, o, where: mask);
                Assert.IsTrue(np.array_equal(o, np.where(mask, np.add(sv.copy(), sv2.copy()), prior)));
            }
        }

        [TestMethod]
        public void Where_NarrowRows_MaskedOffElementsKeepPriorContents_AndIntegerDivideIsSafe()
        {
            const int rows = 100, w = 3;
            var a = np.arange(rows * 2 * w, dtype: np.int32).reshape(rows, 2 * w)[":, :3"] + 1;
            var b = np.zeros(new Shape(rows, w), np.int32);                             // zero divisors everywhere...
            var mask = np.zeros(new Shape(rows, w), np.@bool);                          // ...but nothing is written
            var o = np.full(new Shape(rows, w), -7, np.int32);
            np.floor_divide(a, b, o, where: mask);
            Assert.IsTrue(np.array_equal(o, np.full(new Shape(rows, w), -7, np.int32)));
        }

        [TestMethod]
        public void Where_MaskShorterThanAVectorAndStridedMask_StillCorrect()
        {
            var a = np.arange(20, dtype: np.float64);
            var b = np.full(new Shape(20), 1.0, np.float64);
            var o = np.full(new Shape(20), -1.0, np.float64);
            var mask = (np.floor_divide(np.arange(20), 5) % 2) == 0;                           // 20 bytes: below a 32-byte block
            np.add(a, b, o, where: mask);
            Doubles(o).Should().Equal(Enumerable.Range(0, 20).Select(i => (i / 5 % 2 == 0) ? i + 1.0 : -1.0).ToArray());

            var wide = (np.arange(40) % 4) == 0;
            var strided = wide["::2"];                                          // non-unit mask stride → scalar scan
            var o2 = np.full(new Shape(20), -1.0, np.float64);
            np.add(a, b, o2, where: strided);
            Doubles(o2).Should().Equal(Enumerable.Range(0, 20).Select(i => (i % 2 == 0) ? i + 1.0 : -1.0).ToArray());
        }
    }
}
