using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests following https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.vstack.html
    /// </summary>
    [TestClass]
    public class np_vstack_tests
    {
        [TestMethod]
        public void VStackNDArrays()
        {
            //1D
            var n1 = np.array(new double[] {1, 2, 3});
            var n2 = np.array(new double[] {2, 3, 4});
            var n3 = np.array(new double[] {3, 4, 5});

            var n = np.vstack(n1, n2).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size));

            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[1, 2] == 4);

            //2D
            n1 = np.array(new double[][] {new double[] {1}, new double[] {2}, new double[] {3}});
            n2 = np.array(new double[][] {new double[] {4}, new double[] {5}, new double[] {6}});

            n = np.vstack(n1, n2).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size));

            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[2, 0] == 3);
            Assert.IsTrue(n[3, 0] == 4);
            Assert.IsTrue(n[4, 0] == 5);
            Assert.IsTrue(n[5, 0] == 6);

            //3D
            n1 = np.array(new double[][] {new double[] {1}, new double[] {2}, new double[] {3}});
            n2 = np.array(new double[][] {new double[] {4}, new double[] {5}, new double[] {6}});
            n3 = np.array(new double[][] {new double[] {7}, new double[] {8}, new double[] {9}});

            n = np.vstack(n1, n2, n3).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size + n3.size));

            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[2, 0] == 3);
            Assert.IsTrue(n[3, 0] == 4);
            Assert.IsTrue(n[4, 0] == 5);
            Assert.IsTrue(n[5, 0] == 6);
            Assert.IsTrue(n[6, 0] == 7);
            Assert.IsTrue(n[7, 0] == 8);
            Assert.IsTrue(n[8, 0] == 9);
        }        
    }
}
