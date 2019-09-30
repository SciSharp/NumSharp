using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class np_outer_test
    {
        [TestMethod]
        public void Case1()
        {
            np.outer(np.ones(5), np.linspace(-2, 2, 5))
                .Should().BeOfValues(-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2)
                .And.BeShaped(5, 5);
        }
    }
}
