namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Pins <see cref="NDArray.ToJaggedArray{T}"/> on NON-contiguous views: every rank must read the LOGICAL
    ///     C-order elements through the view's strides and offset.
    /// </summary>
    /// <remarks>
    ///     The rank-2 path used to densify <c>Storage.GetData&lt;T&gt;()</c> and index that buffer as if it were
    ///     C-contiguous: a transposed view returned the WRONG values (row 0 read <c>0, 5, 10</c> instead of
    ///     <c>0, 4, 8</c>) and a column slice read OUT OF BOUNDS (garbage such as <c>-3.0e-241</c>), and the
    ///     densified temp itself leaked for every view (found by the leak audit's catalogue). It now reads each
    ///     element by coordinate. Expected values are NumPy's <c>tolist()</c> of the same views.
    /// </remarks>
    [TestClass]
    public class NDArrayToJaggedArrayViewsTests
    {
        /// <summary>
        ///     A transposed view's rows are the base's columns. NumPy 2.4.2:
        ///     <c>arange(12.).reshape(3,4).T.tolist()</c> → <c>[[0,4,8],[1,5,9],[2,6,10],[3,7,11]]</c>.
        /// </summary>
        [TestMethod]
        public void Rank2_TransposedView_ReadsTheLogicalRows()
        {
            using var m = np.arange(12.0).reshape(3, 4);
            using var t = m.T;

            var j = (double[][])t.ToJaggedArray<double>();

            j.Length.Should().Be(4);
            j[0].Should().Equal(0.0, 4.0, 8.0);
            j[1].Should().Equal(1.0, 5.0, 9.0);
            j[2].Should().Equal(2.0, 6.0, 10.0);
            j[3].Should().Equal(3.0, 7.0, 11.0);
        }

        /// <summary>
        ///     A column slice reads inside its window (offset 1, row stride 4). NumPy 2.4.2:
        ///     <c>arange(12.).reshape(3,4)[:, 1:3].tolist()</c> → <c>[[1,2],[5,6],[9,10]]</c>.
        /// </summary>
        [TestMethod]
        public void Rank2_ColumnSlice_ReadsInsideTheWindow()
        {
            using var m = np.arange(12.0).reshape(3, 4);
            using var s = m[":, 1:3"];

            var j = (double[][])s.ToJaggedArray<double>();

            j.Length.Should().Be(3);
            j[0].Should().Equal(1.0, 2.0);
            j[1].Should().Equal(5.0, 6.0);
            j[2].Should().Equal(9.0, 10.0);
        }

        /// <summary>
        ///     A reversed, stepped 1-D view reads in its own logical order. NumPy 2.4.2:
        ///     <c>arange(10.)[::-3].tolist()</c> → <c>[9, 6, 3, 0]</c>.
        /// </summary>
        [TestMethod]
        public void Rank1_ReversedSteppedView_ReadsTheLogicalOrder()
        {
            using var v = np.arange(10.0);
            using var r = v["::-3"];

            ((double[])r.ToJaggedArray<double>()).Should().Equal(9.0, 6.0, 3.0, 0.0);
        }

        /// <summary>
        ///     A permuted rank-3 view reads every element by coordinate. NumPy 2.4.2, with
        ///     <c>t = arange(24).reshape(2,3,4).transpose(2,0,1)</c>: <c>t[k][i][j] == 12*i + 4*j + k</c>.
        /// </summary>
        [TestMethod]
        public void Rank3_PermutedView_ReadsEveryElementByCoordinate()
        {
            using var a = np.arange(24).reshape(2, 3, 4);
            using var t = a.transpose(new[] { 2, 0, 1 });

            var j = (long[][][])t.ToJaggedArray<long>();

            j.Length.Should().Be(4);
            for (int k = 0; k < 4; k++)
            for (int i = 0; i < 2; i++)
            for (int jj = 0; jj < 3; jj++)
                j[k][i][jj].Should().Be(12L * i + 4L * jj + k);
        }
    }
}
