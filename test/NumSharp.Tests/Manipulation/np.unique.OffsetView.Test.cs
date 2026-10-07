using System.Linq;
using NumSharp.Backends;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     <c>np.unique</c> over a C-contiguous VIEW whose shape carries a non-zero offset into its storage (the layout of
    ///     a view rebuilt over a buffer, as the oracle's replay and <c>np.ndarray(buffer=, offset=)</c> build them). The
    ///     hash path and the key extraction of the <c>return_index</c> / <c>return_inverse</c> / <c>return_counts</c> path
    ///     read from the storage's base address and dropped the offset, so they took the elements BEFORE the view and lost
    ///     its last ones. Expected values probed against NumPy 2.4.2: <c>buf = [5, 3, 9, 3, 1, 9, 7]; np.unique(buf[2:6],
    ///     return_index=True, return_inverse=True, return_counts=True)</c> is <c>([1, 3, 9], [2, 1, 0], [2, 1, 0, 2],
    ///     [1, 1, 2])</c>.
    /// </summary>
    [TestClass]
    public class NpUniqueOffsetViewTest
    {
        /// <summary>
        ///     The four elements <c>[9, 3, 1, 9]</c> of <c>[5, 3, 9, 3, 1, 9, 7]</c> in <paramref name="dtype"/>, as a
        ///     contiguous view that keeps the storage's base address and an offset of 2 (not a re-seated slice).
        /// </summary>
        private static NDArray OffsetView(NPTypeCode dtype)
        {
            var full = np.array(new long[] { 5, 3, 9, 3, 1, 9, 7 }).astype(dtype);
            var view = new NDArray(full.Storage, new Shape(new long[] { 4 }, new long[] { 1 }, 2, 7));
            view.Shape.IsContiguous.Should().BeTrue();
            view.Shape.offset.Should().Be(2);
            return view;
        }

        private static readonly NPTypeCode[] Numeric =
        {
            NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
            NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
            NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        [TestMethod]
        public void Unique_OfAnOffsetView_ReadsTheViewsElements()
        {
            foreach (var dtype in Numeric)
            {
                var r = np.unique(OffsetView(dtype)).values;
                r.astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(new long[] { 1, 3, 9 }, dtype.ToString());
            }
        }

        [TestMethod]
        public void Unique_WithEveryOutput_OfAnOffsetView()
        {
            foreach (var dtype in Numeric)
            {
                var r = np.unique(OffsetView(dtype), return_index: true, return_inverse: true, return_counts: true);
                r.values.astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(new long[] { 1, 3, 9 }, dtype.ToString());
                r.indices.ToArray<long>().Should().Equal(new long[] { 2, 1, 0 }, dtype.ToString());
                r.inverse_indices.ToArray<long>().Should().Equal(new long[] { 2, 1, 0, 2 }, dtype.ToString());
                r.counts.ToArray<long>().Should().Equal(new long[] { 1, 1, 2 }, dtype.ToString());
            }
        }

        [TestMethod]
        public void Unique_OfABooleanOffsetView()
        {
            // buf = [T, F, F, T, F, T, T]; np.unique(buf[2:6], return_index=True, return_inverse=True, return_counts=True)
            // -> ([False, True], [0, 1], [0, 1, 0, 1], [2, 2]). The base-address read gave index [1, 0], inverse [1, 0, 0, 1].
            var full = np.array(new[] { true, false, false, true, false, true, true });
            var view = new NDArray(full.Storage, new Shape(new long[] { 4 }, new long[] { 1 }, 2, 7));
            var r = np.unique(view, return_index: true, return_inverse: true, return_counts: true);
            r.values.ToArray<bool>().Should().Equal(false, true);
            r.indices.ToArray<long>().Should().Equal(0L, 1L);
            r.inverse_indices.ToArray<long>().Should().Equal(0L, 1L, 0L, 1L);
            r.counts.ToArray<long>().Should().Equal(2L, 2L);
        }
    }
}
