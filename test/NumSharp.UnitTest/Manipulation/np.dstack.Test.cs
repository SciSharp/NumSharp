using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests following https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html
    /// </summary>
    [TestClass]
    public class np_dstack_tests
    {
        [TestMethod]
        public void DStackNDArrays()
        {
            //1D
            var n1 = np.array(new double[] {1, 2, 3});
            var n2 = np.array(new double[] {2, 3, 4});

            var n = np.dstack(n1, n2).MakeGeneric<double>();
            n.Should().BeOfSize(n1.size + n2.size).And.BeOfValues(1, 2, 2, 3, 3, 4).And.BeShaped(1, 3, 2);

            //2D
            n = np.dstack(n1.reshape(3, 1), n2.reshape(3, 1)).MakeGeneric<double>();
            n.Should().BeOfSize(n1.size + n2.size).And.BeOfValues(1, 2, 2, 3, 3, 4).And.BeShaped(3, 1, 2);
        }
    }
}
