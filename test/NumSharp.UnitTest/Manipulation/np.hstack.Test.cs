using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests following https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.hstack.html
    /// </summary>
    [TestClass]
    public class np_hstack_tests
    {
        [TestMethod]
        public void HStackNDArrays()
        {
            //1D
            var n1 = np.array(new double[] { 1, 2, 3 });
            var n2 = np.array(new double[] { 2, 3, 4 });

            var n = np.hstack(n1, n2).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size));

            Assert.IsTrue(n[0] == 1);
            Assert.IsTrue(n[1] == 2);
            Assert.IsTrue(n[3] == 2);
            Assert.IsTrue(n[5] == 4);

            ////2D
            n1 = np.array(new double[][] { new double[] { 1 }, new double[] { 2 }, new double[] { 3 } });
            n2 = np.array(new double[][] { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });
            var n3 = np.array(new double[][] { new double[] { 7 }, new double[] { 8 }, new double[] { 9 } });


            n = np.hstack(n1, n2, n3).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size + n3.size));

            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[2, 0] == 3);
            Assert.IsTrue(n[0, 1] == 4);
            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 1] == 6);

            Assert.IsTrue(n[0, 2] == 7);

            var nn = np.hstack(n1, n2).MakeGeneric<double>();
            nn = nn.hstack(n3).MakeGeneric<double>();

            Assert.IsTrue(nn[0, 0] == 1);
            Assert.IsTrue(nn[1, 0] == 2);
            Assert.IsTrue(nn[2, 0] == 3);
            Assert.IsTrue(nn[0, 1] == 4);
            Assert.IsTrue(nn[1, 1] == 5);
            Assert.IsTrue(nn[2, 1] == 6);

            Assert.IsTrue(nn[0, 2] == 7);
            //3D
            n1 = np.array(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }).reshape(2, 3, 2);
            n2 = np.array(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }).reshape(2, 3, 2);

            n = np.hstack(n1, n2).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size));

            Assert.IsTrue(n[0, 0, 0] == 0);
            Assert.IsTrue(n[0, 1, 0] == 2);
            Assert.IsTrue(n[0, 2, 1] == 5);

            ////4D
            n1 = np.arange(24 * 100).reshape(2, 30, 20, 2);
            n2 = np.arange(24 * 100).reshape(2, 30, 20, 2);
            var n4 = np.hstack(n1, n2).MakeGeneric<int>();

            Assert.IsTrue(n4.size == (n1.size + n2.size));
            Assert.IsTrue(n4[0, 0, 0, 0] == 0);
            Assert.IsTrue(n4[0, 0, 0, 1] == 1);
        }
    }
}
