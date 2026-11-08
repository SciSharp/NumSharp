using System;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     np.delete's out-of-bounds texts as NumPy writes them, found when the np-layer dtype plan's P3 batch 7 moved its
    ///     helpers into <c>NDArray.FunctionBase.cs</c> and replayed the snapshot against NumPy 2.4.2: several indices are the
    ///     fancy index of <c>keep[obj,] = False</c>, a 1-D mask, so the error names its axis 0 whatever the deletion axis;
    ///     one index (a lone integer, or an integer list of one) is the single-value path, whose error names the deletion
    ///     axis. Every expectation is NumPy 2.4.2's output.
    /// </summary>
    [TestClass]
    public class np_delete_numpy_order_tests
    {
        private static NDArray A => np.arange(6).reshape(2, 3);

        /// <summary>Several indices: the keep mask's axis 0.</summary>
        [TestMethod]
        public void SeveralIndices_OutOfBounds_NameTheMasksAxisZero()
        {
            // np.delete(np.arange(6).reshape(2, 3), [0, 3], axis=1)
            Action ints = () => np.delete(A, new[] { 0, 3 }, 1);
            ints.Should().ThrowExactly<IndexOutOfRangeException>()
                .WithMessage("index 3 is out of bounds for axis 0 with size 3");

            Action longs = () => np.delete(A, new long[] { 0, 3 }, 1);
            longs.Should().ThrowExactly<IndexOutOfRangeException>()
                .WithMessage("index 3 is out of bounds for axis 0 with size 3");

            // np.delete(np.arange(6).reshape(2, 3), np.array([3, 0]), axis=1)
            Action array = () => np.delete(A, np.array(new long[] { 3, 0 }), 1);
            array.Should().ThrowExactly<IndexOutOfRangeException>()
                .WithMessage("index 3 is out of bounds for axis 0 with size 3");
        }

        /// <summary>One index in a list is NumPy's single-value path: the deletion axis, the index as given.</summary>
        [TestMethod]
        public void OneIndexInAList_OutOfBounds_NamesTheDeletionAxis()
        {
            // np.delete(np.arange(6).reshape(2, 3), [3], axis=1)
            Action ints = () => np.delete(A, new[] { 3 }, 1);
            ints.Should().ThrowExactly<IndexOutOfRangeException>()
                .WithMessage("index 3 is out of bounds for axis 1 with size 3");

            // np.delete(np.arange(6).reshape(2, 3), [-4], axis=1)
            Action longs = () => np.delete(A, new long[] { -4 }, 1);
            longs.Should().ThrowExactly<IndexOutOfRangeException>()
                .WithMessage("index -4 is out of bounds for axis 1 with size 3");
        }

        /// <summary>One index in a list deletes like the lone integer.</summary>
        [TestMethod]
        public void OneIndexInAList_DeletesThatPosition()
        {
            // np.delete(np.arange(6).reshape(2, 3), [1], axis=1) -> [[0, 2], [3, 5]]
            var r = np.delete(A, new[] { 1 }, 1);
            r.shape.Should().BeEquivalentTo(new long[] { 2, 2 });
            r.ToArray<long>().Should().Equal(0L, 2L, 3L, 5L);
        }
    }
}
