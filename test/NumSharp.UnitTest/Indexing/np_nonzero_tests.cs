using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing
{
    [TestClass]
    public class np_nonzero_tests : TestClass
    {
        [TestMethod]
        public void Case1()
        {
            var x = np.array(3, 0, 0, 0, 4, 0, 5, 6, 0).reshape(3, 3);
            var ret = np.nonzero(x);
            ret[0].Should().BeOfValues(0, 1, 2, 2);
            ret[1].Should().BeOfValues(0, 1, 0, 1);
            x[np.nonzero(x)].Should().BeOfValues(3, 4, 5, 6).And.BeShaped(4);
        }

        [TestMethod]
        public void Case2()
        {
            var x = np.arange(9).reshape(3, 3);
            var ret = np.nonzero(x);
            ret[0].Should().BeOfValues(0,0,1,1,1,2,2,2);
            ret[1].Should().BeOfValues(1,2,0,1,2,0,1,2);
        }

    }
}
