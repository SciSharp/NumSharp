using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class NDArray_unique_Test : TestClass
    {
        [TestMethod]
        public void Case1()
        {
            arange(10).unique()
                .Should().BeShaped(10).And.BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Case2()
        {
            np.repeat(arange(10), 10).reshape(10,10).unique()
                .Should().BeShaped(10).And.BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Case2_Sliced()
        {
            var arr = np.repeat(arange(10), 10).reshape(10, 10)[":, 0"];
            Console.WriteLine((string)arr);
            Console.WriteLine(arr.Shape);
            arr.unique().Should().BeShaped(10).And.BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
            Console.WriteLine((string)arr.unique());
        }

        [TestMethod]
        public void Unique_ReturnsSorted_UnsortedInput()
        {
            // NumPy always returns sorted unique values
            // >>> np.unique(np.array([5, 2, 9, 2, 5, 1, 8]))
            // array([1, 2, 5, 8, 9])
            var arr = np.array(new int[] { 5, 2, 9, 2, 5, 1, 8 });
            arr.unique().Should().BeShaped(5).And.BeOfValues(1, 2, 5, 8, 9);
        }

        [TestMethod]
        public void Unique_ReturnsSorted_FloatInput()
        {
            // Test with floats
            var arr = np.array(new double[] { 3.14, 1.41, 2.71, 1.41, 3.14 });
            arr.unique().Should().BeShaped(3).And.BeOfValues(1.41, 2.71, 3.14);
        }

        [TestMethod]
        public void Unique_ReturnsSorted_NegativeValues()
        {
            // Test with negative values
            var arr = np.array(new int[] { -5, 3, -1, 0, 3, -5, 7 });
            arr.unique().Should().BeShaped(5).And.BeOfValues(-5, -1, 0, 3, 7);
        }

        // ============================================================
        //  Keyword-argument tests (NumPy 2.x parity)
        // ============================================================

        [TestMethod]
        public void Unique_ReturnIndex_Basic()
        {
            // >>> u, idx = np.unique([5,2,9,2,5,1,8], return_index=True)
            // u=[1,2,5,8,9]  idx=[5,1,0,6,2]
            var arr = np.array(new int[] { 5, 2, 9, 2, 5, 1, 8 });
            var r = np.unique(arr, return_index: true);

            r.Length.Should().Be(2);
            r[0].Should().BeShaped(5).And.BeOfValues(1, 2, 5, 8, 9);
            r[1].dtype.Should().Be(typeof(long));
            r[1].Should().BeShaped(5);
            r[1].GetInt64(0).Should().Be(5);
            r[1].GetInt64(1).Should().Be(1);
            r[1].GetInt64(2).Should().Be(0);
            r[1].GetInt64(3).Should().Be(6);
            r[1].GetInt64(4).Should().Be(2);
        }

        [TestMethod]
        public void Unique_ReturnInverse_Basic()
        {
            // >>> u, inv = np.unique([5,2,9,2,5,1,8], return_inverse=True)
            // u=[1,2,5,8,9]  inv=[2,1,4,1,2,0,3]
            var arr = np.array(new int[] { 5, 2, 9, 2, 5, 1, 8 });
            var r = np.unique(arr, return_index: false, return_inverse: true);

            r.Length.Should().Be(2);
            r[0].Should().BeShaped(5).And.BeOfValues(1, 2, 5, 8, 9);
            r[1].dtype.Should().Be(typeof(long));
            r[1].Should().BeShaped(7);
            r[1].GetInt64(0).Should().Be(2);
            r[1].GetInt64(1).Should().Be(1);
            r[1].GetInt64(2).Should().Be(4);
            r[1].GetInt64(3).Should().Be(1);
            r[1].GetInt64(4).Should().Be(2);
            r[1].GetInt64(5).Should().Be(0);
            r[1].GetInt64(6).Should().Be(3);
        }

        [TestMethod]
        public void Unique_ReturnCounts_Basic()
        {
            // >>> u, cnt = np.unique([1,2,1,3,2,1], return_counts=True)
            // u=[1,2,3]  cnt=[3,2,1]
            var arr = np.array(new int[] { 1, 2, 1, 3, 2, 1 });
            var r = np.unique(arr, return_index: false, return_counts: true);

            r.Length.Should().Be(2);
            r[0].Should().BeShaped(3).And.BeOfValues(1, 2, 3);
            r[1].dtype.Should().Be(typeof(long));
            r[1].Should().BeShaped(3);
            r[1].GetInt64(0).Should().Be(3);
            r[1].GetInt64(1).Should().Be(2);
            r[1].GetInt64(2).Should().Be(1);
        }

        [TestMethod]
        public void Unique_AllFourReturns()
        {
            // >>> u, idx, inv, cnt = np.unique([1,2,1,3,2,1], return_index=True, return_inverse=True, return_counts=True)
            // u=[1,2,3]  idx=[0,1,3]  inv=[0,1,0,2,1,0]  cnt=[3,2,1]
            var arr = np.array(new int[] { 1, 2, 1, 3, 2, 1 });
            var r = np.unique(arr, return_index: true, return_inverse: true, return_counts: true);

            r.Length.Should().Be(4);
            r[0].Should().BeShaped(3).And.BeOfValues(1, 2, 3);
            r[1].GetInt64(0).Should().Be(0);
            r[1].GetInt64(1).Should().Be(1);
            r[1].GetInt64(2).Should().Be(3);
            r[2].GetInt64(0).Should().Be(0);
            r[2].GetInt64(1).Should().Be(1);
            r[2].GetInt64(2).Should().Be(0);
            r[2].GetInt64(3).Should().Be(2);
            r[2].GetInt64(4).Should().Be(1);
            r[2].GetInt64(5).Should().Be(0);
            r[3].GetInt64(0).Should().Be(3);
            r[3].GetInt64(1).Should().Be(2);
            r[3].GetInt64(2).Should().Be(1);
        }

        [TestMethod]
        public void Unique_2D_NoAxis_InverseSameShape()
        {
            // >>> a = np.array([[5,2],[2,5],[1,8]])
            // >>> u, idx, inv, cnt = np.unique(a, return_index=True, return_inverse=True, return_counts=True)
            // u=[1,2,5,8]  idx=[4,1,0,5]  inv.shape=(3,2)  inv=[[2,1],[1,2],[0,3]]  cnt=[1,2,2,1]
            var arr = np.array(new int[] { 5, 2, 2, 5, 1, 8 }).reshape(3, 2);
            var r = np.unique(arr, return_index: true, return_inverse: true, return_counts: true);

            r.Length.Should().Be(4);
            r[0].Should().BeShaped(4).And.BeOfValues(1, 2, 5, 8);
            r[1].GetInt64(0).Should().Be(4);
            r[1].GetInt64(1).Should().Be(1);
            r[1].GetInt64(2).Should().Be(0);
            r[1].GetInt64(3).Should().Be(5);

            // Inverse should be same shape as input (NumPy 2.x behavior)
            r[2].shape.Should().BeEquivalentTo(new long[] { 3, 2 });
            r[2][0, 0].GetInt64(0).Should().Be(2);
            r[2][0, 1].GetInt64(0).Should().Be(1);
            r[2][1, 0].GetInt64(0).Should().Be(1);
            r[2][1, 1].GetInt64(0).Should().Be(2);
            r[2][2, 0].GetInt64(0).Should().Be(0);
            r[2][2, 1].GetInt64(0).Should().Be(3);

            r[3].GetInt64(0).Should().Be(1);
            r[3].GetInt64(1).Should().Be(2);
            r[3].GetInt64(2).Should().Be(2);
            r[3].GetInt64(3).Should().Be(1);
        }

        [TestMethod]
        public void Unique_EqualNan_True()
        {
            // equal_nan=True (default): single NaN in output
            // >>> u, cnt = np.unique([nan, 1.0, nan, 2.0, 1.0], return_counts=True)
            // u=[1, 2, nan]  cnt=[2, 1, 2]
            var arr = np.array(new double[] { double.NaN, 1.0, double.NaN, 2.0, 1.0 });
            var r = np.unique(arr, return_index: true, return_inverse: true, return_counts: true);

            r[0].Should().BeShaped(3);
            r[0].GetDouble(0).Should().Be(1.0);
            r[0].GetDouble(1).Should().Be(2.0);
            double.IsNaN(r[0].GetDouble(2)).Should().BeTrue();

            // idx=[1, 3, 0]
            r[1].GetInt64(0).Should().Be(1);
            r[1].GetInt64(1).Should().Be(3);
            r[1].GetInt64(2).Should().Be(0);

            // inv=[2, 0, 2, 1, 0]
            r[2].GetInt64(0).Should().Be(2);
            r[2].GetInt64(1).Should().Be(0);
            r[2].GetInt64(2).Should().Be(2);
            r[2].GetInt64(3).Should().Be(1);
            r[2].GetInt64(4).Should().Be(0);

            // cnt=[2, 1, 2]
            r[3].GetInt64(0).Should().Be(2);
            r[3].GetInt64(1).Should().Be(1);
            r[3].GetInt64(2).Should().Be(2);
        }

        [TestMethod]
        public void Unique_EqualNan_False()
        {
            // equal_nan=False: each NaN is unique
            // >>> u, idx, inv, cnt = np.unique([nan, 1.0, nan, 2.0, 1.0], return_index=True, return_inverse=True, return_counts=True, equal_nan=False)
            // u=[1, 2, nan, nan]  idx=[1, 3, 0, 2]  inv=[2, 0, 3, 1, 0]  cnt=[2, 1, 1, 1]
            var arr = np.array(new double[] { double.NaN, 1.0, double.NaN, 2.0, 1.0 });
            var r = np.unique(arr, return_index: true, return_inverse: true, return_counts: true, equal_nan: false);

            r[0].Should().BeShaped(4);
            r[0].GetDouble(0).Should().Be(1.0);
            r[0].GetDouble(1).Should().Be(2.0);
            double.IsNaN(r[0].GetDouble(2)).Should().BeTrue();
            double.IsNaN(r[0].GetDouble(3)).Should().BeTrue();

            r[1].GetInt64(0).Should().Be(1);
            r[1].GetInt64(1).Should().Be(3);
            r[1].GetInt64(2).Should().Be(0);
            r[1].GetInt64(3).Should().Be(2);

            r[2].GetInt64(0).Should().Be(2);
            r[2].GetInt64(1).Should().Be(0);
            r[2].GetInt64(2).Should().Be(3);
            r[2].GetInt64(3).Should().Be(1);
            r[2].GetInt64(4).Should().Be(0);

            r[3].GetInt64(0).Should().Be(2);
            r[3].GetInt64(1).Should().Be(1);
            r[3].GetInt64(2).Should().Be(1);
            r[3].GetInt64(3).Should().Be(1);
        }

        [TestMethod]
        public void Unique_Axis0_RowDedup()
        {
            // >>> a = np.array([[1,2],[1,2],[3,4],[5,6]])
            // >>> u, idx, inv, cnt = np.unique(a, axis=0, return_index=True, return_inverse=True, return_counts=True)
            // u=[[1,2],[3,4],[5,6]]  idx=[0,2,3]  inv=[0,0,1,2]  cnt=[2,1,1]
            var arr = np.array(new int[] { 1, 2, 1, 2, 3, 4, 5, 6 }).reshape(4, 2);
            var r = np.unique(arr, return_index: true, return_inverse: true, return_counts: true, axis: 0);

            r.Length.Should().Be(4);
            r[0].shape.Should().BeEquivalentTo(new long[] { 3, 2 });
            r[0][0, 0].GetInt32(0).Should().Be(1);
            r[0][0, 1].GetInt32(0).Should().Be(2);
            r[0][1, 0].GetInt32(0).Should().Be(3);
            r[0][1, 1].GetInt32(0).Should().Be(4);
            r[0][2, 0].GetInt32(0).Should().Be(5);
            r[0][2, 1].GetInt32(0).Should().Be(6);

            r[1].GetInt64(0).Should().Be(0);
            r[1].GetInt64(1).Should().Be(2);
            r[1].GetInt64(2).Should().Be(3);

            r[2].GetInt64(0).Should().Be(0);
            r[2].GetInt64(1).Should().Be(0);
            r[2].GetInt64(2).Should().Be(1);
            r[2].GetInt64(3).Should().Be(2);

            r[3].GetInt64(0).Should().Be(2);
            r[3].GetInt64(1).Should().Be(1);
            r[3].GetInt64(2).Should().Be(1);
        }

        [TestMethod]
        public void Unique_Axis1_ColumnDedup()
        {
            // >>> a = np.array([[1,2,1],[3,4,3]])
            // >>> np.unique(a, axis=1)
            // array([[1, 2], [3, 4]])
            var arr = np.array(new int[] { 1, 2, 1, 3, 4, 3 }).reshape(2, 3);
            var r = np.unique(arr, return_index: false, axis: 1);

            r.Length.Should().Be(1);
            r[0].shape.Should().BeEquivalentTo(new long[] { 2, 2 });
            r[0][0, 0].GetInt32(0).Should().Be(1);
            r[0][0, 1].GetInt32(0).Should().Be(2);
            r[0][1, 0].GetInt32(0).Should().Be(3);
            r[0][1, 1].GetInt32(0).Should().Be(4);
        }

        [TestMethod]
        public void Unique_NegativeAxis()
        {
            // >>> a = np.array([[1,2],[1,2],[3,4]])
            // axis=-2 (=axis=0) → [[1,2],[3,4]]
            var arr = np.array(new int[] { 1, 2, 1, 2, 3, 4 }).reshape(3, 2);
            var r = np.unique(arr, return_index: false, axis: -2);

            r[0].shape.Should().BeEquivalentTo(new long[] { 2, 2 });
            r[0][0, 0].GetInt32(0).Should().Be(1);
            r[0][0, 1].GetInt32(0).Should().Be(2);
            r[0][1, 0].GetInt32(0).Should().Be(3);
            r[0][1, 1].GetInt32(0).Should().Be(4);
        }

        [TestMethod]
        public void Unique_AxisOutOfRange_Throws()
        {
            var arr = np.array(new int[] { 1, 2, 3 });
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() =>
                np.unique(arr, return_index: false, axis: 1));
        }

        [TestMethod]
        public void Unique_EmptyArray_AllReturns()
        {
            // >>> u, idx, inv, cnt = np.unique([], return_index=True, return_inverse=True, return_counts=True)
            // all empty arrays
            var arr = np.array(new int[0]);
            var r = np.unique(arr, return_index: true, return_inverse: true, return_counts: true);

            r[0].size.Should().Be(0);
            r[1].size.Should().Be(0);
            r[2].size.Should().Be(0);
            r[3].size.Should().Be(0);
        }

        [TestMethod]
        public void Unique_Int64Input_IndexDtypeIsInt64()
        {
            // NumPy returns int64 indices regardless of input dtype
            var arr = np.array(new long[] { 1L, 2L, 1L });
            var r = np.unique(arr, return_index: true, return_inverse: true, return_counts: true);

            r[1].dtype.Should().Be(typeof(long));
            r[2].dtype.Should().Be(typeof(long));
            r[3].dtype.Should().Be(typeof(long));
        }

        [TestMethod]
        public void Unique_Float_NaN_Reconstruct()
        {
            // Verify a[idx] == u and u[inv] == a (with NaN-aware semantics)
            var arr = np.array(new double[] { 3.0, double.NaN, 1.0, 2.0, 1.0, double.NaN });
            var r = np.unique(arr, return_index: true, return_inverse: true);

            // Reconstruct via idx: each value at u[i] should equal arr[idx[i]]
            for (int i = 0; i < r[0].size; i++)
            {
                var uVal = r[0].GetDouble(i);
                var idxVal = (int)r[1].GetInt64(i);
                var aVal = arr.GetDouble(idxVal);
                if (double.IsNaN(uVal))
                    double.IsNaN(aVal).Should().BeTrue($"arr[idx[{i}]] should be NaN");
                else
                    aVal.Should().Be(uVal, $"arr[idx[{i}]] should equal u[{i}]");
            }

            // Reconstruct via inv: u[inv[i]] should equal arr[i]
            for (int i = 0; i < arr.size; i++)
            {
                var invVal = (int)r[2].GetInt64(i);
                var uVal = r[0].GetDouble(invVal);
                var aVal = arr.GetDouble(i);
                if (double.IsNaN(uVal))
                    double.IsNaN(aVal).Should().BeTrue($"u[inv[{i}]] should be NaN");
                else
                    aVal.Should().Be(uVal, $"u[inv[{i}]] should equal arr[{i}]");
            }
        }

        [TestMethod]
        public void Unique_AllBoolFlagCombinations_ReturnCountMatches()
        {
            var arr = np.array(new int[] { 1, 2, 1, 3 });

            // 8 combinations of 3 flags
            for (int mask = 0; mask < 8; mask++)
            {
                bool ri = (mask & 1) != 0;
                bool rv = (mask & 2) != 0;
                bool rc = (mask & 4) != 0;
                int expectedCount = 1 + (ri ? 1 : 0) + (rv ? 1 : 0) + (rc ? 1 : 0);
                var r = np.unique(arr, return_index: ri, return_inverse: rv, return_counts: rc);
                r.Length.Should().Be(expectedCount, $"mask={mask}");
                r[0].Should().BeShaped(3); // unique = [1,2,3]
            }
        }

        [TestMethod]
        public void Unique_HasAllKeywordArguments()
        {
            // Audit: verify all NumPy keyword arguments are present on np.unique
            var methods = typeof(np).GetMethods()
                .Where(m => m.Name == "unique")
                .ToList();
            methods.Should().NotBeEmpty();

            var allParamNames = methods
                .SelectMany(m => m.GetParameters().Select(p => p.Name))
                .Distinct()
                .ToList();

            allParamNames.Should().Contain("return_index");
            allParamNames.Should().Contain("return_inverse");
            allParamNames.Should().Contain("return_counts");
            allParamNames.Should().Contain("axis");
            allParamNames.Should().Contain("equal_nan");
        }
    }
}
