using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class np_prod_tests
    {

        [TestMethod]
        public void EmptyArray()
        {
            np.prod(np.array(new int[0])).Should().BeScalar(1);
        }

        [TestMethod]
        public void Case1()
        {
            var a = np.prod(np.array(1f, 2f));
            a.GetValue(0).Should().Be(2);
            a.Shape.IsScalar.Should().BeTrue();
        }

        [TestMethod]
        public void Case1_double()
        {
            var a = np.prod(np.array(1d, 2d));
            a.GetValue(0).Should().Be(2d);
            a.Shape.IsScalar.Should().BeTrue();
        }

        [TestMethod]
        public void Case2()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2));
            a.GetValue(0).Should().Be(24f);
            a.Shape.IsScalar.Should().BeTrue();
        }

        [TestMethod]
        public void Case3()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2), axis: 1);
            a.GetValue(0).Should().Be(2f);
            a.GetValue(1).Should().Be(12f);
            a.shape.Should().HaveCount(1).And.ContainInOrder(2);
        }

        [TestMethod]
        public void Case4()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2), axis: 1);
            a.GetValue(0).Should().Be(2f);
            a.GetValue(1).Should().Be(12f);
            a.shape.Should().HaveCount(1).And.ContainInOrder(2);
        }

        [TestMethod]
        public void Case4_dtype()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2), axis: 1, dtype: np.uint8);
            a.GetValue(0).Should().Be(2f);
            a.GetValue(1).Should().Be(12f);
            a.shape.Should().HaveCount(1).And.ContainInOrder(2);
            a.dtype.Should().Be<byte>();
        }
    }
}
