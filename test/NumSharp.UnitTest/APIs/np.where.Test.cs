using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class NpWhereTest
    {
        [TestMethod]
        public void WhereWithConditionOnly()
        {
            var condition = np.array(new bool[] { true, false, true, false });
            var result = np.where(condition);
            
            Assert.AreEqual(1, result.Length); // Should return one array of indices
            Assert.AreEqual(2, result[0].size); // Two true values
            Assert.IsTrue(NDArray.Equals(result[0], np.array(new int[] { 0, 2 }))); // Indices should be [0, 2]
        }

        [TestMethod]
        public void WhereWith3Args1D()
        {
            var condition = np.array(new bool[] { true, false, true, false });
            var x = np.array(new int[] { 1, 2, 3, 4 });
            var y = np.array(new int[] { 10, 20, 30, 40 });
            
            var result = np.where(condition, x, y);
            
            Assert.IsTrue(NDArray.Equals(result, np.array(new int[] { 1, 20, 3, 40 })));
        }

        [TestMethod]
        public void WhereWith3Args2D()
        {
            var condition = np.array(new bool[,] { { true, false }, { true, true } });
            var x = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var y = np.array(new int[,] { { 9, 8 }, { 7, 6 } });
            
            var result = np.where(condition, x, y);
            
            Assert.IsTrue(NDArray.Equals(result, np.array(new int[,] { { 1, 8 }, { 3, 4 } })));
        }

        [TestMethod]
        public void WhereWithBroadcasting()
        {
            var x = np.arange(3).reshape(3, 1);
            var y = np.arange(4).reshape(1, 4);
            var condition = x < y;
            
            var result = np.where(condition, x, 10 + y);
            
            // Expected: when x < y, take from x, otherwise from 10 + y
            var expected = np.array(new int[,] { 
                { 10, 0, 0, 0 }, 
                { 10, 11, 1, 1 }, 
                { 10, 11, 12, 2 } 
            });
            
            Assert.IsTrue(NDArray.Equals(result, expected));
        }

        [TestMethod]
        public void WhereWithScalarBroadcasting()
        {
            var a = np.array(new int[,] { { 0, 1, 2 }, { 0, 2, 4 }, { 0, 3, 6 } });
            var result = np.where(a < 4, a, -1);
            
            var expected = np.array(new int[,] { 
                { 0, 1, 2 }, 
                { 0, 2, -1 }, 
                { 0, 3, -1 } 
            });
            
            Assert.IsTrue(NDArray.Equals(result, expected));
        }
    }
}
