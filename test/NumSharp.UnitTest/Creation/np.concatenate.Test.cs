using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_concatenate_test
    {
        [TestMethod]
        public void Case1_Axis1()
        {
            var a = np.full(1, (3, 1, 3), NPTypeCode.Int32);
            var b = np.full(2, (3, 2, 3), NPTypeCode.Int32);
            var c = np.concatenate(1, a, b);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, 0, :"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, 1, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, 2, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis1_Cast()
        {
            var a = np.full(1, (3, 1, 3), NPTypeCode.Int32);
            var b = np.full(2, (3, 2, 3), NPTypeCode.Double);
            var c = np.concatenate(1, a, b);

            c.dtype.Should().Be<double>();
            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, 0, :"].flat.Cast<double>().Should().AllBeEquivalentTo(1);
            c[":, 1, :"].flat.Cast<double>().Should().AllBeEquivalentTo(2);
            c[":, 2, :"].flat.Cast<double>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis0()
        {
            var a = np.full(1, (1, 3, 3), NPTypeCode.Int32);
            var b = np.full(2, (2, 3, 3), NPTypeCode.Int32);
            var c = np.concatenate(0, a, b);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c["0, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c["1, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c["2, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis2()
        {
            var a = np.full(1, (3, 3, 1), NPTypeCode.Int32);
            var b = np.full(2, (3, 3, 2), NPTypeCode.Int32);
            var c = np.concatenate(2, a, b);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, :, 0"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, :, 1"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, :, 2"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis_minus1()
        {
            var a = np.full(1, (3, 3, 1), NPTypeCode.Int32);
            var b = np.full(2, (3, 3, 2), NPTypeCode.Int32);
            var c = np.concatenate(-1, a, b);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, :, 0"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, :, 1"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, :, 2"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case2_Axis1_3Arrays_Cast()
        {
            var a = np.full(1, (3, 1, 3), NPTypeCode.Int32);
            var b = np.full(2, (3, 2, 3), NPTypeCode.Decimal);
            var c = np.full(2, (3, 1, 3), NPTypeCode.Byte);
            var d = np.concatenate(1, a, b, c);
            d.dtype.Should().Be<decimal>();
            d.shape.Should().HaveCount(3).And.ContainInOrder(3, 4, 3);
            d.size.Should().Be(36);
            d[":, 0, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(1);
            d[":, 1, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(2);
            d[":, 2, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(2);
        }
    }
}
