using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class np_clip_test
    {
        [TestMethod]
        public void Case1()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, 3, max).Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8).And.BeShaped(3, 4);
        }

        [TestMethod]
        public void Case2()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, max, null).Should().BeOfValues(8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11).And.BeShaped(3, 4);
        }

        [TestMethod]
        public void Case3()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, null, max).Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8).And.BeShaped(3, 4);
        }
    }
}
