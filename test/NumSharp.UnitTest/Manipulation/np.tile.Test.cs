using System;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class TileTest
    {
        [TestMethod]
        public void Tile_1DArray_SingleRepeat()
        {
            var arr = np.arange(3); // [0, 1, 2]
            var tiled = np.tile(arr, new int[] { 3 });
            
            var expected = np.array(new int[] {0, 1, 2, 0, 1, 2, 0, 1, 2});
            
            Assert.AreEqual(expected.shape, tiled.shape);
            Assert.IsTrue(np.array_equal(expected, tiled));
        }

        [TestMethod]
        public void Tile_2DArray_RowRepeat()
        {
            var arr = np.array(new double[,] {{1, 2}, {3, 4}});
            var tiled = np.tile(arr, new int[] { 2, 1 });
            
            var expected = np.array(new double[,] {{1, 2}, {3, 4}, {1, 2}, {3, 4}});
            
            Assert.AreEqual(expected.shape, tiled.shape);
            Assert.IsTrue(np.array_equal(expected, tiled));
        }

        [TestMethod]
        public void Tile_2DArray_ColumnRepeat()
        {
            var arr = np.array(new double[,] {{1, 2}, {3, 4}});
            var tiled = np.tile(arr, new int[] { 1, 2 });
            
            var expected = np.array(new double[,] {{1, 2, 1, 2}, {3, 4, 3, 4}});
            
            Assert.AreEqual(expected.shape, tiled.shape);
            Assert.IsTrue(np.array_equal(expected, tiled));
        }

        [TestMethod]
        public void Tile_2DArray_BothRepeat()
        {
            var arr = np.array(new double[,] {{1, 2}, {3, 4}});
            var tiled = np.tile(arr, new int[] { 2, 2 });
            
            var expected = np.array(new double[,] 
            {
                {1, 2, 1, 2}, 
                {3, 4, 3, 4}, 
                {1, 2, 1, 2}, 
                {3, 4, 3, 4}
            });
            
            Assert.AreEqual(expected.shape, tiled.shape);
            Assert.IsTrue(np.array_equal(expected, tiled));
        }

        [TestMethod]
        public void Tile_2DArray_LessRepeats()
        {
            var arr = np.array(new double[,] {{1, 2}, {3, 4}});
            var tiled = np.tile(arr, new int[] { 2 });
            
            var expected = np.array(new double[,] {{1, 2, 1, 2}, {3, 4, 3, 4}});
            
            Assert.AreEqual(expected.shape, tiled.shape);
            Assert.IsTrue(np.array_equal(expected, tiled));
        }

        [TestMethod]
        public void Tile_Scalar()
        {
            var arr = np.array(5);
            var tiled = np.tile(arr, new int[] { 3, 2 });
            
            var expected = np.array(new double[,] {{5, 5}, {5, 5}, {5, 5}});
            
            Assert.AreEqual(expected.shape, tiled.shape);
            Assert.IsTrue(np.array_equal(expected, tiled));
        }
    }
}
