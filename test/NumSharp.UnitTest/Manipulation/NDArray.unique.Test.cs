using System;
using System.Linq;
using FluentAssertions;
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
    }
}
