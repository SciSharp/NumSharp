using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_repeat_tests
    {
        [TestMethod]
        public void Scalar()
        {
            var nd = np.repeat(3, 4);
            Assert.AreEqual(nd.Data<int>().Count(x => x == 3), 4);
        }

        [TestMethod]
        public void Simple2DArray()
        {
            var x = np.array(new int[][] {new int[] {1, 2}, new int[] {3, 4}});

            var nd = np.repeat(x, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] {1, 1, 2, 2, 3, 3, 4, 4}, nd.Data<int>()));
        }
    }
}
