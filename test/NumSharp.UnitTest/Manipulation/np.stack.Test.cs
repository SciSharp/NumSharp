using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests following https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
    /// </summary>
    [TestClass]
    public class np_stack_tests
    {
        [TestMethod]
        public void Case1()
        {
            //1D
            var n1 = np.array(new double[] {1, 2, 3});
            var n2 = np.array(new double[] {2, 3, 4});

            var n = np.stack(new NDArray[]{ n1, n2 }, 0).MakeGeneric<double>();
            n.Should().BeOfSize((int)(n1.size + n2.size)).And.BeOfValues(1, 2, 3, 2, 3, 4).And.BeShaped(2, 3);

            //2D
            n = np.stack(new NDArray[]{ n1, n2 }, 1).MakeGeneric<double>();
            n.Should().BeOfSize((int)(n1.size + n2.size)).And.BeOfValues(1, 2, 2, 3, 3, 4).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Stack_EmptyArrays_AddsNewAxis()
        {
            // Regression: stack builds on np.expand_dims, which used to no-op on empty inputs, so
            // stacking empties concatenated (wrong) instead of adding a new axis. NumPy adds an axis
            // of size = number of arrays, regardless of emptiness.
            np.stack(new[] { np.zeros(new[] {0}), np.zeros(new[] {0}) }).Should().BeShaped(2, 0);         // (0,)  -> (2,0)
            np.stack(new[] { np.zeros(new[] {3, 0}), np.zeros(new[] {3, 0}) }).Should().BeShaped(2, 3, 0); // (3,0) -> (2,3,0)
            np.stack(new[] { np.zeros(new[] {2, 0, 3}), np.zeros(new[] {2, 0, 3}) }).Should().BeShaped(2, 2, 0, 3);
        }

        [TestMethod]
        public void VStack_Empty1D_Promotes()
        {
            // vstack routes 1-D through atleast_2d -> expand_dims; the empty 1-D case was broken.
            // NumPy: vstack([(0,), (0,)]) -> (2, 0).
            np.vstack(new[] { np.zeros(new[] {0}), np.zeros(new[] {0}) }).Should().BeShaped(2, 0);
        }
    }
}
