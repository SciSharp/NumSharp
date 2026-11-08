namespace NumSharp.Tests.Selection
{
    [TestClass]
    public class NDArrayAMinTest
    {
        [TestMethod]
        public void amin()
        {
            //no axis
            var n = np.arange(4).reshape(2, 2);
            var n1 = np.amin(n).MakeGeneric<long>();
            // A reduction without an axis is 0-d: read the scalar itself (indexing it is NumPy's IndexError).
            Assert.IsTrue((long)n1 == 0);

            //2D with axis
            n1 = np.amin(n, 0).MakeGeneric<long>();
            Assert.IsTrue((long)n1[0] == 0);
            Assert.IsTrue((long)n1[1] == 1);
            n1 = np.amin(n, 1).MakeGeneric<long>();
            Assert.IsTrue((long)n1[0] == 0);
            Assert.IsTrue((long)n1[1] == 2);

            //3D
            n = np.arange(24).reshape(4, 3, 2);
            n1 = np.amin(n, 0).MakeGeneric<long>();
            Assert.IsTrue((long)n1[0, 1] == 1);
            Assert.IsTrue((long)n1[2, 1] == 5);
            Assert.IsTrue((long)n1[1, 1] == 3);
            n1 = np.amin(n, 1).MakeGeneric<long>();
            Assert.IsTrue((long)n1[1, 1] == 7);
            Assert.IsTrue((long)n1[2, 1] == 13);
            Assert.IsTrue((long)n1[3, 0] == 18);

            //4D
            n = np.arange(24).reshape(2, 3, 2, 2);
            n1 = np.amin(n, 1).MakeGeneric<long>();
            Assert.IsTrue((long)n1[0, 0, 1] == 1);
            Assert.IsTrue((long)n1[1, 0, 1] == 13);
            Assert.IsTrue((long)n1[1, 1, 1] == 15);
            n1 = np.amin(n, 3).MakeGeneric<long>();
            Assert.IsTrue((long)n1[0, 1, 1] == 6);
            Assert.IsTrue((long)n1[1, 1, 1] == 18);
            Assert.IsTrue((long)n1[1, 2, 1] == 22);
        }
    }
}
