using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class np_all_axis_Test
    {
        [TestMethod]
        public void np_all_axis_2D()
        {
            // Test array: [[true, false, true], [true, true, true]]
            var arr = np.array(new bool[,] { { true, false, true }, { true, true, true } });
            
            // Test axis=0 (along columns): should be [true, false, true] (all in each column)
            var result_axis0 = np.all(arr, axis: 0);
            var expected_axis0 = np.array(new bool[] { true, false, true });
            Assert.IsTrue(np.array_equal(result_axis0, expected_axis0));

            // Test axis=1 (along rows): should be [false, true] (all in each row)
            var result_axis1 = np.all(arr, axis: 1);
            var expected_axis1 = np.array(new bool[] { false, true });
            Assert.IsTrue(np.array_equal(result_axis1, expected_axis1));
        }

        [TestMethod]
        public void np_all_axis_3D()
        {
            // Create a 3D array for testing
            var arr = np.ones(new int[] { 2, 3, 4 }); // All ones (truthy)
            arr[0, 1, 2] = 0; // Add one falsy value

            // Test different axes
            var result_axis0 = np.all(arr, axis: 0); // Shape should be (3, 4)
            Assert.AreEqual(2, result_axis0.ndim);
            Assert.AreEqual(3, result_axis0.shape[0]);
            Assert.AreEqual(4, result_axis0.shape[1]);

            var result_axis1 = np.all(arr, axis: 1); // Shape should be (2, 4)
            Assert.AreEqual(2, result_axis1.ndim);
            Assert.AreEqual(2, result_axis1.shape[0]);
            Assert.AreEqual(4, result_axis1.shape[1]);

            var result_axis2 = np.all(arr, axis: 2); // Shape should be (2, 3)
            Assert.AreEqual(2, result_axis2.ndim);
            Assert.AreEqual(2, result_axis2.shape[0]);
            Assert.AreEqual(3, result_axis2.shape[1]);
        }

        [TestMethod]
        public void np_all_keepdims()
        {
            var arr = np.array(new bool[,] { { true, false, true }, { true, true, true } });

            // Test with keepdims=true
            var result_keepdims = np.all(arr, axis: 0, keepdims: true);
            Assert.AreEqual(2, result_keepdims.ndim); // Should maintain original number of dimensions
            Assert.AreEqual(1, result_keepdims.shape[0]); // The reduced axis becomes size 1
            Assert.AreEqual(3, result_keepdims.shape[1]); // Other dimensions remain

            var result_keepdims1 = np.all(arr, axis: 1, keepdims: true);
            Assert.AreEqual(2, result_keepdims1.ndim); // Should maintain original number of dimensions
            Assert.AreEqual(2, result_keepdims1.shape[0]); // Other dimensions remain
            Assert.AreEqual(1, result_keepdims1.shape[1]); // The reduced axis becomes size 1
        }

        [TestMethod]
        public void np_all_different_types()
        {
            // Test with integer array
            var int_arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 0, 6 } }); // Contains a zero (falsy value)
            var int_result = np.all(int_arr, axis: 1);
            // First row: all non-zero -> true, Second row: contains zero -> false
            Assert.AreEqual(true, int_result[0]);
            Assert.AreEqual(false, int_result[1]);
        }
    }
}