using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
using np = NumSharp.Core.NumPy;

namespace NumSharp.UnitTest.Selection
{
    [TestClass]
    public class IndexingTest
    {
        [TestMethod]
        public void IndexAccessorGetter()
        {
            var nd = np.arange(12).reshape(3, 4);

            Assert.IsTrue(nd.Data<int>(1, 1) == 5);
            Assert.IsTrue(nd.Data<int>(2, 0) == 8);

        }

        [TestMethod]
        public void IndexAccessorSetter()
        {
            var nd = np.arange(12).reshape(3, 4);

            Assert.IsTrue(nd.Data<int>(0, 3) == 3);
            Assert.IsTrue(nd.Data<int>(1, 3) == 7);

            // set value
            nd.Set(new Shape(0), 10);
            Assert.IsTrue(nd.Data<int>(0, 0) == 10);
            Assert.IsTrue(nd.Data<int>(0, 3) == 10);
            Assert.IsTrue(nd.Data<int>(1, 3) == 7);
        }
    }
}
