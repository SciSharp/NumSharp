using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Statistics
{
    [TestClass]
    public class np_var_tests
    {
        [TestMethod]
        public void Case1()
        {
            var nd1 = np.arange(4).reshape(2, 2);
            nd1.var().Data<double>()[0].Should().BeApproximately(1.25D, 0.0001);
        }

        [TestMethod]
        public void Case2()
        {
            var a = np.zeros((2, 4 * 4), dtype: np.float32);
            a["0, :"] = 1.0;
            a["1, :"] = 0.1;
            var ret = np.var(a);
            ret.GetValue<float>(0).Should().BeApproximately(0.2025F, 0.001f);
        }

        [TestMethod]
        public void Case3()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.var(0);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case4()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.var(1);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>().Should().AllBeEquivalentTo(0.5d * 0.5d);
        }

        [TestMethod]
        public void Case5()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.var(0, ddof: 1);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>()[0].Should().BeApproximately(2, 0.0001d);
            ret.ToArray<double>()[1].Should().BeApproximately(2, 0.0001d);
        }

        [TestMethod]
        public void Case6()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.var(1, ddof: 1);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>()[0].Should().BeApproximately(0.5, 0.0001d);
            ret.ToArray<double>()[1].Should().BeApproximately(0.5, 0.0001d);
        }

        [TestMethod]
        public void Case7()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.var(ddof: 1);
            ret.Shape.IsScalar.Should().BeTrue();
            ret.ToArray<double>()[0].Should().BeApproximately(1.66666666666667D, 0.0001d);
        }
    }
}
