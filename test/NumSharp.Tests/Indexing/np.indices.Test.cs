using System;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     Tests for <see cref="np.indices"/> / <see cref="np.indices_sparse"/> — focused on the
    ///     NumSharp-friendly dimension spellings (int[] / long[] / Shape). Expected values are from
    ///     NumPy 2.4.2 (<c>np.indices((2,3))</c> etc.); the differential-fuzz corpus already gates
    ///     the full value matrix, so these pin the API SURFACE: that the house long/Shape shape
    ///     types bind with no down-cast and produce results identical to the legacy int[] path.
    /// </summary>
    [TestClass]
    public class np_indices_Test
    {
        // ─────────────────────────────────────────────────────────── dense: spelling parity

        [TestMethod]
        public void Indices_KnownValues_2x3()
        {
            // numpy: grid[0] = row indices [[0,0,0],[1,1,1]], grid[1] = col indices [[0,1,2],[0,1,2]].
            var g = np.indices(new long[] { 2, 3 });
            string.Join(",", g.Shape.Dimensions).Should().Be("2,2,3");
            g[0].ToArray<long>().Should().Equal(0L, 0, 0, 1, 1, 1);
            g[1].ToArray<long>().Should().Equal(0L, 1, 2, 0, 1, 2);
        }

        [TestMethod]
        public void Indices_LongOverload_EqualsIntPath()
        {
            var viaInt = np.indices(new int[] { 2, 3, 4 });
            var viaLong = np.indices(new long[] { 2, 3, 4 });
            viaLong.dtype.Should().Be(viaInt.dtype);
            string.Join(",", viaLong.Shape.Dimensions).Should().Be(string.Join(",", viaInt.Shape.Dimensions));
            np.array_equal(viaInt, viaLong).Should().BeTrue();
        }

        [TestMethod]
        public void Indices_ShapeTupleOverload_ReadsLikePython()
        {
            // np.indices((2,3)) — the C# value tuple converts implicitly to Shape.
            var viaTuple = np.indices((2, 3));
            var viaInt = np.indices(new int[] { 2, 3 });
            np.array_equal(viaInt, viaTuple).Should().BeTrue();
        }

        [TestMethod]
        public void Indices_ArraysOwnShape_BindsLongOverload_NoDownCast()
        {
            // The friction case the fix targets: NDArray.shape is long[], so it must reach
            // np.indices with no manual (int) narrowing.
            var arr = np.zeros(new Shape(3, 4));
            var g = np.indices(arr.shape);          // arr.shape is long[]
            string.Join(",", g.Shape.Dimensions).Should().Be("2,3,4");
            np.array_equal(g, np.indices(new int[] { 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void Indices_ShapeStructOverload()
        {
            var arr = np.zeros(new Shape(3, 4));
            var g = np.indices(arr.Shape);          // arr.Shape is a Shape
            np.array_equal(g, np.indices(new int[] { 3, 4 })).Should().BeTrue();
        }

        // ─────────────────────────────────────────────────────────── dense: dtype + edges

        [TestMethod]
        public void Indices_LongOverload_RespectsDtype()
        {
            var i32 = np.indices(new long[] { 2, 3 }, NPTypeCode.Int32);
            i32.dtype.Should().Be(NPTypeCode.Int32);
            np.array_equal(i32.astype(NPTypeCode.Int64), np.indices(new long[] { 2, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Indices_EmptyDimensions_ReturnsZeroShape()
        {
            // numpy: np.indices(()) -> shape (0,).
            string.Join(",", np.indices(new long[0]).Shape.Dimensions).Should().Be("0");
            string.Join(",", np.indices(new int[0]).Shape.Dimensions).Should().Be("0");
        }

        [TestMethod]
        public void Indices_NegativeDimension_Throws()
        {
            Action act = () => np.indices(new long[] { 2, -1 });
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Indices_DimensionBeyondInt32_IsNotTruncated()
        {
            // A dimension > int.MaxValue paired with a 0 has size 0 (allocation-free), but the
            // shape must carry the full 64-bit value. The old int[]-only API could not even
            // express this, and a silent (int) narrowing would flip it negative and throw.
            long big = (long)int.MaxValue + 1; // 2147483648
            var g = np.indices(new long[] { big, 0 });
            g.size.Should().Be(0);
            string.Join(",", g.Shape.Dimensions).Should().Be("2," + big + ",0");
        }

        // ─────────────────────────────────────────────────────────── sparse: spelling parity

        [TestMethod]
        public void IndicesSparse_LongOverload_EqualsIntPath()
        {
            var viaInt = np.indices_sparse(new int[] { 2, 3 });
            var viaLong = np.indices_sparse(new long[] { 2, 3 });
            viaLong.Length.Should().Be(viaInt.Length);
            // axis-d array has shape (1,…,dim,…,1): (2,1) and (1,3).
            string.Join(",", viaLong[0].Shape.Dimensions).Should().Be("2,1");
            string.Join(",", viaLong[1].Shape.Dimensions).Should().Be("1,3");
            np.array_equal(viaInt[0], viaLong[0]).Should().BeTrue();
            np.array_equal(viaInt[1], viaLong[1]).Should().BeTrue();
        }

        [TestMethod]
        public void IndicesSparse_ShapeTupleOverload()
        {
            var viaTuple = np.indices_sparse((2, 3));
            var viaInt = np.indices_sparse(new int[] { 2, 3 });
            viaTuple.Length.Should().Be(2);
            np.array_equal(viaTuple[0], viaInt[0]).Should().BeTrue();
            np.array_equal(viaTuple[1], viaInt[1]).Should().BeTrue();
        }

        [TestMethod]
        public void IndicesSparse_EmptyDimensions_ReturnsEmptyTuple()
        {
            np.indices_sparse(new long[0]).Should().BeEmpty();
        }

        // ─────────────────────────────────────────────────────────── downstream consumers

        [TestMethod]
        public void Mgrid_StillProducesCorrectGrid_AfterLongIndicesRefactor()
        {
            // mgrid feeds indices a long[] internally now (was int[] via a checked cast).
            var (x, y) = np.mgrid["0:2", "0:3"];
            x.ToArray<long>().Should().Equal(0L, 0, 0, 1, 1, 1);
            y.ToArray<long>().Should().Equal(0L, 1, 2, 0, 1, 2);
        }

        [TestMethod]
        public void MaIndices_LongAndShapeOverloads()
        {
            string.Join(",", np.ma.indices(new long[] { 2, 3 }).filled().Shape.Dimensions).Should().Be("2,2,3");
            string.Join(",", np.ma.indices((2, 3)).filled().Shape.Dimensions).Should().Be("2,2,3");
        }

        // ─────────────────────────────────────────────────────────── .NET-friendly idioms

        [TestMethod]
        public void Indices_CollectionExpression_BindsIntAndLong()
        {
            // A C# 12 collection expression of int literals binds int[] (no ambiguity with the
            // long[] overload — an identity element conversion beats the implicit int->long one),
            // while long literals bind long[]. Both must compile AND agree.
            np.array_equal(np.indices([2, 3]), np.indices(new int[] { 2, 3 })).Should().BeTrue();
            np.array_equal(np.indices([2L, 3L]), np.indices(new long[] { 2, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Indices_RowColUnpackIdiom()
        {
            // NumPy's `row, col = np.indices((2, 3))` unpacks the leading axis; in C# the dense
            // result is one NDArray indexed by the leading axis.
            var g = np.indices((2, 3));
            var row = g[0];
            var col = g[1];
            row.ToArray<long>().Should().Equal(0L, 0, 0, 1, 1, 1);
            col.ToArray<long>().Should().Equal(0L, 1, 2, 0, 1, 2);
        }

        [TestMethod]
        public void Indices_ResultIsChainableNDArray()
        {
            // The result is an ordinary NDArray, so it flows into the fluent surface.
            var flat = np.indices((2, 3))[1].reshape(6L);
            flat.ToArray<long>().Should().Equal(0L, 1, 2, 0, 1, 2);
        }

        [TestMethod]
        public void IndicesSparse_ForeachAndLinq()
        {
            // The sparse result is a plain NDArray[] — foreach- and LINQ-able.
            var sp = np.indices_sparse((2, 3));
            System.Linq.Enumerable.Select(sp, a => string.Join("x", a.Shape.Dimensions))
                .Should().Equal("2x1", "1x3");

            long total = 0;
            foreach (var a in sp) total += a.size;
            total.Should().Be(5); // 2 + 3
        }

        [TestMethod]
        public void Indices_DtypeSpellings_AllNetForms()
        {
            // dtype accepts every .NET spelling NumSharp offers — a Type, an NPTypeCode, a dtype
            // string, or a DType descriptor — all resolving to the same int32 grid.
            np.indices((2, 2), typeof(int)).dtype.Should().Be(NPTypeCode.Int32);
            np.indices((2, 2), NPTypeCode.Int32).dtype.Should().Be(NPTypeCode.Int32);
            np.indices((2, 2), "i4").dtype.Should().Be(NPTypeCode.Int32);
            np.indices((2, 2), np.int32).dtype.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Indices_ScalarConvenience_And1DSpellings()
        {
            // Documented NumSharp convenience (Shape overload): a scalar converts implicitly to a
            // 1-D Shape, so np.indices(5) is the (1, 5) grid. NumPy's np.indices(5) raises (an int
            // is not a sequence) — this is a deliberate, documented divergence.
            var s = np.indices(5);
            string.Join(",", s.Shape.Dimensions).Should().Be("1,5");
            s.ToArray<long>().Should().Equal(0L, 1, 2, 3, 4);

            // C# has no 1-element tuple, so a 1-D grid is spelled [5] or new long[]{5} (both agree).
            np.array_equal(np.indices([5]), s).Should().BeTrue();
            np.array_equal(np.indices(new long[] { 5 }), s).Should().BeTrue();
        }
    }
}
