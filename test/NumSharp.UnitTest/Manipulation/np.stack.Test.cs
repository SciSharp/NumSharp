using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests following https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html
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
            n.Should().BeOfSize(n1.size + n2.size).And.BeOfValues(1, 2, 3, 2, 3, 4).And.BeShaped(2, 3);

            //2D
            n = np.stack(new NDArray[]{ n1, n2 }, 1).MakeGeneric<double>();
            n.Should().BeOfSize(n1.size + n2.size).And.BeOfValues(1, 2, 2, 3, 3, 4).And.BeShaped(3, 2);
        }
    }
}
