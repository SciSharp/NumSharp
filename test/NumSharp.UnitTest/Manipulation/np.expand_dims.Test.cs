using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class ExpandDimsTest
    {
        [TestMethod]
        public void Simple1DArrayTo2DArray()
        {
            var input = np.array(1, 2, 3);
            var expected = np.array(new int[] {1, 2, 3});

            var result = np.expand_dims(input, 0);

            Assert.IsTrue(result.shape[0] == 1);
            Assert.IsTrue(result.shape[1] == 3);
            Assert.IsTrue(result.ndim == 2);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<int>(), expected.Data<int>()));
        }

        [TestMethod]
        public void Simple1DArrayToTransposed2D()
        {
            var input = np.array(1, 2, 3);
            var expected = np.array(new int[] {1, 2, 3});

            var result = np.expand_dims(input, 1);

            Assert.IsTrue(result.shape[0] == 3);
            Assert.IsTrue(result.shape[1] == 1);
            Assert.IsTrue(result.ndim == 2);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<int>(), expected.Data<int>()));
        }

        [TestMethod]
        public void Simple1DArrayToTransposed3D()
        {
            var input = np.array(1, 2, 3);
            var expected = np.array(new int[] {1, 2, 3});

            var result = np.expand_dims(np.expand_dims(input, 1), 2);

            Assert.IsTrue(result.shape[0] == 3);
            Assert.IsTrue(result.shape[1] == 1);
            Assert.IsTrue(result.shape[2] == 1);
            Assert.IsTrue(result.ndim == 3);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<int>(), expected.Data<int>()));
        }
    }
}
