using System;
using System.IO;
using NumSharp;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/absolute-basics.md</c> ("NumSharp: the absolute basics for
    ///     beginners", the NumSharp conversion of NumPy's absolute_beginners guide). Each test mirrors
    ///     one documented snippet and asserts the behaviour the page claims, so the page cannot silently
    ///     drift from the library. Observed values were captured by running the snippets against this
    ///     branch (NumPy 2.4.2 parity).
    /// </summary>
    [TestClass]
    public class AbsoluteBasicsDocTests
    {
        private string _dir;

        [TestInitialize]
        public void Setup()
        {
            _dir = Path.Combine(Path.GetTempPath(), "ns_absbasics_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        [TestCleanup]
        public void Cleanup()
        {
            try { Directory.Delete(_dir, recursive: true); }
            catch (IOException) { /* a leaked handle fails the test that leaked it */ }
        }

        private string At(string name) => Path.Combine(_dir, name);

        // ── Array fundamentals ───────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Fundamentals_CreateIndexMutateSliceView()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            ((int)a[0]).Should().Be(1);

            a[0] = 10;
            a.ToArray<int>().Should().Equal(10, 2, 3, 4, 5, 6);

            a["0:3"].ToArray<int>().Should().Equal(10, 2, 3);

            var b = a["3:"];                        // a view
            b[0] = 40;
            a.ToArray<int>().Should().Equal(new[] { 10, 2, 3, 40, 5, 6 }, "the slice is a view — write-through mutates a");

            var m = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
            ((int)m[1, 3]).Should().Be(8);
        }

        // ── Array attributes ─────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Attributes_NdimShapeSizeDtype()
        {
            var a = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
            a.ndim.Should().Be(2);
            a.shape.Should().Equal(new long[] { 3, 4 });
            a.size.Should().Be(12);
            a.typecode.Should().Be(NPTypeCode.Int32, "np.array(int[]) follows the .NET int type (NumPy would be int64)");
        }

        // ── How to create a basic array ──────────────────────────────────────────────────────────────

        [TestMethod]
        public void Create_ZerosOnesEmptyArangeLinspace()
        {
            np.zeros(2).ToArray<double>().Should().Equal(0.0, 0.0);
            np.ones(2).ToArray<double>().Should().Equal(1.0, 1.0);
            np.empty(2).size.Should().Be(2);
            np.arange(4).ToArray<long>().Should().Equal(0L, 1, 2, 3);
            np.arange(2, 9, 2).ToArray<long>().Should().Equal(2L, 4, 6, 8);
            np.linspace(0, 10, 5).ToArray<double>().Should().Equal(0.0, 2.5, 5.0, 7.5, 10.0);
            np.ones(2, np.int64).typecode.Should().Be(NPTypeCode.Int64);
        }

        // ── Adding, removing, sorting ────────────────────────────────────────────────────────────────

        [TestMethod]
        public void SortAndConcatenate()
        {
            np.sort(np.array(new[] { 2, 1, 5, 3, 7, 4, 6, 8 })).ToArray<int>().Should().Equal(1, 2, 3, 4, 5, 6, 7, 8);

            var joined = np.concatenate(new[] { np.array(new[] { 1, 2, 3, 4 }), np.array(new[] { 5, 6, 7, 8 }) });
            joined.ToArray<int>().Should().Equal(1, 2, 3, 4, 5, 6, 7, 8);

            var stacked = np.concatenate(new[] { np.array(new[,] { { 1, 2 }, { 3, 4 } }), np.array(new[,] { { 5, 6 } }) }, axis: 0);
            stacked.shape.Should().Equal(new long[] { 3, 2 });
        }

        // ── Shape / size / ndim (3-D) ────────────────────────────────────────────────────────────────

        [TestMethod]
        public void ShapeSizeNdim_3D()
        {
            var e = np.array(new[, ,]
            {
                { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } },
                { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } },
                { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } }
            });
            e.ndim.Should().Be(3);
            e.size.Should().Be(24);
            e.shape.Should().Equal(new long[] { 3, 2, 4 });
        }

        // ── Reshape ──────────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Reshape()
        {
            np.arange(6).reshape(3, 2).shape.Should().Equal(new long[] { 3, 2 });
            np.reshape(np.arange(6), (1, 6)).shape.Should().Equal(new long[] { 1, 6 });
        }

        // ── newaxis / expand_dims ────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void NewAxis_RowAndColumn()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            a[np.newaxis].shape.Should().Equal(new long[] { 1, 6 });                 // row vector
            a[Slice.All, np.newaxis].shape.Should().Equal(new long[] { 6, 1 });      // column vector
            np.expand_dims(a, 1).shape.Should().Equal(new long[] { 6, 1 });
            np.expand_dims(a, 0).shape.Should().Equal(new long[] { 1, 6 });
        }

        // ── Indexing / boolean masks / nonzero ───────────────────────────────────────────────────────

        [TestMethod]
        public void BooleanMasksAndNonzero()
        {
            var a = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
            a[a < 5].ToArray<int>().Should().Equal(1, 2, 3, 4);
            a[a % 2 == 0].ToArray<int>().Should().Equal(2, 4, 6, 8, 10, 12);
            a[(a > 2) & (a < 11)].ToArray<int>().Should().Equal(3, 4, 5, 6, 7, 8, 9, 10);

            var idx = np.nonzero(a < 5);
            idx.Length.Should().Be(2, "one index array per axis");
            a[idx[0], idx[1]].ToArray<int>().Should().Equal(1, 2, 3, 4);
        }

        // ── Create from existing data (stack / split / view / copy) ──────────────────────────────────

        [TestMethod]
        public void Stack_Split_ViewCopy()
        {
            np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 })["3:8"].ToArray<int>().Should().Equal(4, 5, 6, 7, 8);

            var a1 = np.array(new[,] { { 1, 1 }, { 2, 2 } });
            var a2 = np.array(new[,] { { 3, 3 }, { 4, 4 } });
            np.vstack(a1, a2).shape.Should().Equal(new long[] { 4, 2 });
            np.hstack(a1, a2).shape.Should().Equal(new long[] { 2, 4 });

            var x = np.arange(1, 25).reshape(2, 12);
            np.hsplit(x, 3).Length.Should().Be(3);
            np.hsplit(x, new[] { 3, 4 }).Length.Should().Be(3);

            var m = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
            var b1 = m[0];                          // a view
            b1[0] = 99;
            ((int)m[0, 0]).Should().Be(99, "b1 is a view of a's first row");
        }

        // ── Basic array operations ───────────────────────────────────────────────────────────────────

        [TestMethod]
        public void BasicOps_And_SumAxis()
        {
            var data = np.array(new[] { 1, 2 });
            var ones = np.ones(2, np.int32);
            (data + ones).ToArray<int>().Should().Equal(2, 3);
            (data - ones).ToArray<int>().Should().Equal(0, 1);
            (data * data).ToArray<int>().Should().Equal(1, 4);

            var q = data / data;                    // true division → float64
            ((double)q[0]).Should().Be(1.0);
            ((double)q[1]).Should().Be(1.0);

            ((long)np.array(new[] { 1, 2, 3, 4 }).sum()).Should().Be(10L);   // int32 sum widens to int64

            var b = np.array(new[,] { { 1, 1 }, { 2, 2 } });
            b.sum(axis: 0).ToArray<long>().Should().Equal(3L, 3);
            b.sum(axis: 1).ToArray<long>().Should().Equal(2L, 4);
        }

        // ── Broadcasting ─────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Broadcasting_Scalar()
        {
            (np.array(new[] { 1.0, 2.0 }) * 1.6).ToArray<double>().Should().Equal(1.6, 3.2);
        }

        // ── More useful array operations (aggregations + axis) ───────────────────────────────────────

        [TestMethod]
        public void Aggregations_WithAxis()
        {
            var d = np.array(new[] { 1, 2, 3 });
            ((int)d.max()).Should().Be(3);
            ((int)d.min()).Should().Be(1);
            ((long)d.sum()).Should().Be(6L);
            ((double)d.mean()).Should().Be(2.0);
            ((long)d.prod()).Should().Be(6L);
            ((double)d.std()).Should().BeApproximately(0.816496580927726, 1e-12);

            var a = np.array(new[,] { { 1, 2 }, { 5, 3 }, { 4, 6 } });
            a.max(axis: 0).ToArray<int>().Should().Equal(5, 6);
            a.max(axis: 1).ToArray<int>().Should().Equal(2, 5, 6);
            a.min(axis: 0).ToArray<int>().Should().Equal(1, 2);
        }

        // ── Creating matrices ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Matrices_IndexAggregateBroadcast()
        {
            var data = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            ((int)data[0, 1]).Should().Be(2);
            data["1:3"].shape.Should().Equal(new long[] { 2, 2 });
            data["0:2, 0"].ToArray<int>().Should().Equal(1, 3);

            var row = np.array(new[,] { { 1, 1 } });
            (data + row).ToArray<int>().Should().Equal(2, 3, 4, 5, 6, 7);
        }

        // ── Generating random numbers ────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Random_SeededDeterministic()
        {
            np.random.seed(42);
            var r1 = np.random.rand(2, 3);
            np.random.seed(42);
            var r2 = np.random.rand(2, 3);
            np.array_equal(r1, r2).Should().BeTrue("the same seed reproduces the same sequence");
            r1.shape.Should().Equal(new long[] { 2, 3 });

            var ri = np.random.randint(0, 5, (2, 4));
            ri.shape.Should().Equal(new long[] { 2, 4 });
            ((long)np.max(ri) < 5).Should().BeTrue("high is exclusive");
            ((long)np.min(ri) >= 0).Should().BeTrue("low is inclusive");
        }

        // ── Unique items and counts ──────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Unique_ValuesIndexCounts_And_Axis()
        {
            var a = np.array(new[] { 11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20 });
            np.unique(a).values.ToArray<int>().Should().Equal(11, 12, 13, 14, 15, 16, 17, 18, 19, 20);

            var (values, indices) = np.unique(a, return_index: true);
            indices.ToArray<long>().Should().Equal(0L, 2, 3, 4, 5, 6, 7, 12, 13, 14);

            var (vals, counts) = np.unique(a, return_counts: true);
            counts.ToArray<long>().Should().Equal(3L, 2, 2, 2, 1, 1, 1, 1, 1, 1);

            var a2d = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 1, 2, 3, 4 } });
            np.unique(a2d, axis: 0).values.shape.Should().Equal(new long[] { 3, 4 });
        }

        // ── Transposing ──────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void TransposeAndT()
        {
            var arr = np.arange(6).reshape(2, 3);
            arr.T.ToArray<long>().Should().Equal(0L, 3, 1, 4, 2, 5);
            arr.transpose().shape.Should().Equal(new long[] { 3, 2 });
        }

        // ── Reversing (flip) ─────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Flip_And_InPlaceColumn()
        {
            np.flip(np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8 })).ToArray<int>().Should().Equal(8, 7, 6, 5, 4, 3, 2, 1);

            var m = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
            np.flip(m, axis: 0).ToArray<int>().Should().Equal(9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4);
            np.flip(m, axis: 1).ToArray<int>().Should().Equal(4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9);

            var c = np.array(new[,] { { 1, 2, 3, 4 }, { 8, 7, 6, 5 }, { 9, 10, 11, 12 } });
            c[":, 1"] = np.flip(c[":, 1"]);         // reverse the second column in place
            c[":, 1"].ToArray<int>().Should().Equal(10, 7, 2);
        }

        // ── Flatten vs ravel ─────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void FlattenIsCopy_RavelIsView()
        {
            var x = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });

            var a1 = x.flatten();
            a1[0] = 99;
            ((int)x[0, 0]).Should().Be(1, "flatten returns a copy");

            var a2 = x.ravel();
            a2[0] = 98;
            ((int)x[0, 0]).Should().Be(98, "ravel returns a view");
        }

        // ── Working with mathematical formulas (MSE) ─────────────────────────────────────────────────

        [TestMethod]
        public void MseFormula()
        {
            var predictions = np.array(new[] { 1.0, 1.0, 1.0 });
            var labels = np.array(new[] { 1.0, 2.0, 3.0 });
            ((double)np.mean(np.square(predictions - labels))).Should().BeApproximately(1.6666666666666665, 1e-12);
        }

        // ── Save and load ────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void SaveLoad_Npy_And_Text()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            np.save(At("filename.npy"), a);
            np.load_npy(At("filename.npy")).ToArray<int>().Should().Equal(1, 2, 3, 4, 5, 6);

            var csv = np.array(new[] { 1.0, 2, 3, 4, 5, 6, 7, 8 });
            np.savetxt(At("new_file.csv"), csv);
            np.loadtxt(At("new_file.csv")).ToArray<double>().Should().Equal(1.0, 2, 3, 4, 5, 6, 7, 8);
        }

        // ── Importing and exporting a CSV ────────────────────────────────────────────────────────────

        [TestMethod]
        public void SaveTxt_WithFmtDelimiterHeader()
        {
            var a = np.array(new[,] { { -2.58, 0.43, -1.24, 1.60 }, { 0.99, 1.17, 0.94, -0.15 } });
            np.savetxt(At("np.csv"), a, fmt: "%.2f", delimiter: ",", header: "1,2,3,4");
            var text = File.ReadAllText(At("np.csv"));
            text.Should().Contain("# 1,2,3,4");
            text.Should().Contain("-2.58,0.43,-1.24,1.60");
        }

        // ── Plotting (hand values to a .NET chart library) ───────────────────────────────────────────

        [TestMethod]
        public void ToArray_ForPlotting()
        {
            var y = np.array(new[] { 2.0, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22 });
            double[] values = y.ToArray<double>();
            values.Length.Should().Be(13);
            values[0].Should().Be(2.0);
        }
    }
}
