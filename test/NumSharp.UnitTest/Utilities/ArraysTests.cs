using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Utilities
{
    [TestClass]
    public class ArraysTests
    {
        [TestMethod]
        public void Create_1()
        {
            Arrays.Create(typeof(int), 1000).Should().BeOfType<int[]>().And.HaveCount(1000);
        }

        [TestMethod]
        public void Create_2()
        {
            Arrays.Create(typeof(int), new int[] {1000}).Should().BeOfType<int[]>().And.HaveCount(1000);
        }

        [TestMethod]
        public void Create_3()
        {
            Arrays.Create(NPTypeCode.Int32, 1000).Should().BeOfType<int[]>().And.HaveCount(1000);
        }

        [TestMethod]
        public void Create_4()
        {
            Arrays.Create(NPTypeCode.NDArray, 1000).Should().BeOfType<NDArray[]>().And.HaveCount(1000);
        }
    }
}
