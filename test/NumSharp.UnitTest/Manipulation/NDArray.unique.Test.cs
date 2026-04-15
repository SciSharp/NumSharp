using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
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
    }
}
