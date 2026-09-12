using System.Linq;

namespace NumSharp.Tests.Operations {
    [TestClass]
    public class NDArrayNotTest
    {
        [TestMethod]
        public void not_1d()
        {
            var np1 = new NDArray(new[] { false, false, false, false}, new Shape(4));

            var np3 = !np1;

            Assert.IsTrue(Enumerable.SequenceEqual(new[] {true, true, true, true}, np3.Data<bool>().MemoryBlock));
        }

        [TestMethod]
        public void BoolTwo2D_NDArrayOR()
        {
            var np1 = new NDArray(new[] { false, true, false, false }, new Shape(2,2));

            var np3 = !np1;

            Assert.IsTrue(Enumerable.SequenceEqual(new[] { true, false, true, true }, np3.Data<bool>().MemoryBlock));
        }

        // Regression: the raw-buffer fast path in operator ! reads self.Address LINEARLY, which
        // matches logical C-order only for a C-contiguous, offset-0 operand. For an F-contiguous,
        // transposed, strided or sliced-with-offset operand it used to scramble the result (it read
        // the backing buffer in memory order but wrote a fresh C-contiguous output). These must now
        // agree with np.logical_not (the layout-aware ufunc) for every layout.

        [TestMethod]
        public void Not_FContiguous_MatchesLogicalNot()
        {
            var b = np.array(new[] { true, false, true, false, true, false }).reshape(2, 3);
            var bf = np.asfortranarray(b);
            (!bf).ToArray<bool>().Should().Equal(false, true, false, true, false, true);
        }

        [TestMethod]
        public void Not_Transposed_MatchesLogicalNot()
        {
            var b = np.array(new[] { true, false, true, false, true, false }).reshape(2, 3);
            var bt = b.T;                                            // (3,2), non-contiguous
            // bt C-ravel = T,F,F,T,T,F → !bt = F,T,T,F,F,T
            (!bt).ToArray<bool>().Should().Equal(false, true, true, false, false, true);
        }

        [TestMethod]
        public void Not_StridedAndSliced_MatchesLogicalNot()
        {
            var b = np.array(new[] { true, false, true, false, true, false, true, false });
            var strided = b["::2"];                                  // [T,T,T,T]
            (!strided).ToArray<bool>().Should().Equal(false, false, false, false);
            var sliced = b["3:7"];                                   // offset view [F,T,F,T]
            (!sliced).ToArray<bool>().Should().Equal(true, false, true, false);
        }
    }
}