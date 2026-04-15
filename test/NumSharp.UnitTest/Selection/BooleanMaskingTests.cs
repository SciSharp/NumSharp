using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Selection
{
    /// <summary>
    /// Tests for boolean masking based on NumPy 2.4.2 behavior.
    /// </summary>
    [TestClass]
    public class BooleanMaskingTests
    {
        [TestMethod]
        public void ExplicitMask_1D_FiltersCorrectly()
        {
            // NumPy: a[mask] where mask = [True, False, True, False, True, False]
            // Result: [1, 3, 5]
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var mask = np.array(new[] { true, false, true, false, true, false }).MakeGeneric<bool>();
            var result = a[mask];

            result.Should().BeOfValues(1, 3, 5).And.BeShaped(3);
        }

        [TestMethod]
        public void Condition_1D_FiltersCorrectly()
        {
            // NumPy: a[a % 2 == 1] = [1, 3, 5]
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var result = a[a % 2 == 1];

            result.Should().BeOfValues(1, 3, 5).And.BeShaped(3);
        }

        [TestMethod]
        public void ExplicitMask_MatchesCondition()
        {
            // Explicit mask and condition should produce same result
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var mask = np.array(new[] { true, false, true, false, true, false }).MakeGeneric<bool>();

            var explicit_result = a[mask];
            var condition_result = a[a % 2 == 1];

            explicit_result.Should().BeOfValues(1, 3, 5);
            condition_result.Should().BeOfValues(1, 3, 5);
        }

        [TestMethod]
        public void AllTrue_ReturnsAll()
        {
            // NumPy: a[[T,T,T]] = [1, 2, 3]
            var a = np.array(new[] { 1, 2, 3 });
            var mask = np.array(new[] { true, true, true }).MakeGeneric<bool>();
            var result = a[mask];

            result.Should().BeOfValues(1, 2, 3).And.BeShaped(3);
        }

        [TestMethod]
        public void AllFalse_ReturnsEmpty()
        {
            // NumPy: a[[F,F,F]] = [], shape (0,)
            var a = np.array(new[] { 1, 2, 3 });
            var mask = np.array(new[] { false, false, false }).MakeGeneric<bool>();
            var result = a[mask];

            Assert.AreEqual(0, result.size);
            Assert.AreEqual(1, result.ndim);  // 1D empty array
        }

        [TestMethod]
        public void TwoDimensional_RowSelection()
        {
            // NumPy: 2D array with 1D mask selects rows
            // a = [[1,2,3],[4,5,6],[7,8,9]]
            // mask = [True, False, True]
            // result = [[1,2,3],[7,8,9]], shape (2,3)
            var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            var mask = np.array(new[] { true, false, true }).MakeGeneric<bool>();
            var result = a[mask];

            // First row should be [1,2,3], second row [7,8,9]
            Assert.AreEqual(2, result.shape[0]);  // 2 rows selected
            Assert.AreEqual(3, result.shape[1]);  // 3 columns
            result["0"].Should().BeOfValues(1, 2, 3);
            result["1"].Should().BeOfValues(7, 8, 9);
        }

        [TestMethod]
        public void TwoDimensional_ElementMask_Flattens()
        {
            // NumPy: 2D array with 2D mask of same shape selects elements, flattens
            // a = [[1,2],[3,4]]
            // mask = [[True,False],[False,True]]
            // result = [1, 4], shape (2,)
            var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var mask = np.array(new[,] { { true, false }, { false, true } }).MakeGeneric<bool>();
            var result = a[mask];

            result.Should().BeOfValues(1, 4).And.BeShaped(2);
        }

        [TestMethod]
        public void EmptyResult_PreservesDtype()
        {
            // NumPy: Empty result preserves dtype
            var a = np.array(new double[] { 1.0, 2.0, 3.0 });
            var mask = np.array(new[] { false, false, false }).MakeGeneric<bool>();
            var result = a[mask];

            Assert.AreEqual(0, result.size);
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }
    }
}
