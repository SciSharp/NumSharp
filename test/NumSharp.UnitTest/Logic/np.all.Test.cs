using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class np_all_Test
    {
        [TestMethod]
        public void np_all_1D()
        {
            var np1 = new NDArray(new[] {true, true, false, false}, new Shape(4));
            var np2 = new NDArray(typeof(bool), new Shape(150));
            var np3 = new NDArray(new[] {true, true, true, true, true, true, true, true}, new Shape(8));
            Assert.IsFalse(np.all(np1));
            Assert.IsFalse(np.all(np2));
            Assert.IsTrue(np.all(np3));
        }

        [TestMethod]
        public void np_all_2D()
        {
            var np1 = new NDArray(new bool[] {true, true, false, false, true, false}, new Shape(2, 3));
            var np2 = new NDArray(typeof(bool), new Shape(39, 17));
            var np3 = new NDArray(new[] {true, true, true, true, true, true, true, true}, new Shape(2, 4));
            Assert.IsFalse(np.all(np1));
            Assert.IsFalse(np.all(np2));
            Assert.IsTrue(np.all(np3));
        }

        [TestMethod]
        public void np_all_0D_WithAxis0_ReturnsScalar()
        {
            // NumPy 2.x: np.all(0D_array, axis=0) returns 0D boolean scalar
            var arr = np.array(5);  // truthy
            Assert.AreEqual(0, arr.ndim);

            var result = np.all(arr, axis: 0);
            Assert.AreEqual(0, result.ndim, "Result should be 0D");
            Assert.AreEqual(true, (bool)result);

            // Test with falsy value
            var arrFalsy = np.array(0);
            var resultFalsy = np.all(arrFalsy, axis: 0);
            Assert.AreEqual(false, (bool)resultFalsy);
        }

        [TestMethod]
        public void np_all_0D_WithAxisNeg1_ReturnsScalar()
        {
            // NumPy 2.x: np.all(0D_array, axis=-1) is equivalent to axis=0
            var arr = np.array(true);
            var result = np.all(arr, axis: -1);
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(true, (bool)result);
        }

        [TestMethod]
        public void np_all_0D_WithInvalidAxis_Throws()
        {
            // NumPy 2.x: np.all(0D_array, axis=1) raises AxisError
            var arr = np.array(5);
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.all(arr, axis: 1));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.all(arr, axis: -2));
        }
    }
}
