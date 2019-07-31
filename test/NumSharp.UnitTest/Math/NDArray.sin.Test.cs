using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class SinTest
    {
        [TestMethod]
        public void Simple1DArray()
        {
            var nd = np.array(new double[] {0, 30, 45, 60, 90}) * (System.Math.PI / 180);

            var nd2 = np.sin(nd);

            Assert.IsTrue(nd2.GetInt32(0) == 0);
            Assert.IsTrue(nd2.GetInt32(1) < 0.501);
            Assert.IsTrue(nd2.GetInt32(1) > 0.498);
            Assert.IsTrue(nd2.GetInt32(2) < 0.708);
            Assert.IsTrue(nd2.GetInt32(2) > 0.7069);
            Assert.IsTrue(nd2.GetInt32(3) < 0.867);
            Assert.IsTrue(nd2.GetInt32(3) > 0.8659);
            Assert.IsTrue(nd2.GetInt32(4) == 1);
        }
    }
}
