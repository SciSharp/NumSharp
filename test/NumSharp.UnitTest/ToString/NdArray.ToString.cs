using System.Linq;
using System.Text.RegularExpressions;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.ToString
{
    [TestClass]
    public class NdArrayToStringTest : TestClass
    {
        [TestMethod]
        public void Scalar()
        {
            var nd = NDArray.Scalar(3112);
            Assert.AreEqual("3112", nd.ToString());
        }

        [TestMethod]
        public void ArraySmall1Dim()
        {
            var nd = np.array(DataSample.Int32D12);
            Assert.AreEqual("[1, 2, 1, 3, 2, ..., 2, 3, 2, 3, 2]", nd.ToString());
        }

        [TestMethod]
        public void ArrayLarge1Dim()
        {
            var nd = np.arange(10000);
            Assert.AreEqual("[0, 1, 2, 3, 4, ..., 9995, 9996, 9997, 9998, 9999]", nd.ToString());
        }

        [TestMethod]
        public void ArrayLarge2Dim()
        {
            var nd = np.ones(100 * 100).reshape(100, 100);
            Assert.AreEqual(11, Regex.Matches(nd.ToString(), @"\.\.\.").Count);
        }

        [TestMethod]
        public void ArrayLarge3Dim()
        {
            var nd = np.ones(100 * 100 * 100).reshape(100, 100, 100);
            Assert.AreEqual(111, Regex.Matches(nd.ToString(), @"\.\.\.").Count);
        }
    }
}
