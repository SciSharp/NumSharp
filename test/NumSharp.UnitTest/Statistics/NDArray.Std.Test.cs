using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Statistics
{
    public class np_std_tests
    {
        [Test]
        public void Case1()
        {
            var nd1 = np.arange(4).reshape(2, 2);
            nd1.std().Data<double>()[0].Should().BeApproximately(1.118033, 0.0001);
        }

        [Test]
        public void Case2()
        {
            var a = np.zeros((2, 4 * 4), dtype: np.float32);
            a["0, :"] = 1.0;
            a["1, :"] = 0.1;
            var ret = np.std(a);
            ret.GetValue<float>(0).Should().BeApproximately(0.45f, 0.001f);
        }

        [Test]
        public void Case3()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.std(0);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>().Should().AllBeEquivalentTo(1);
        }

        [Test]
        public void Case4()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.std(1);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>().Should().AllBeEquivalentTo(0.5d);
        }

        [Test]
        public void Case5()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.std(0, ddof: 1);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>()[0].Should().BeApproximately(1.41421356, 0.0001d);
            ret.ToArray<double>()[1].Should().BeApproximately(1.41421356, 0.0001d);
        }

        [Test]
        public void Case6()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.std(1, ddof: 1);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
            ret.ToArray<double>()[0].Should().BeApproximately(0.70710678, 0.0001d);
            ret.ToArray<double>()[1].Should().BeApproximately(0.70710678, 0.0001d);
        }

        [Test]
        public void Case7()
        {
            var nd1 = np.arange(4).reshape(2, 2);

            var ret = nd1.std(ddof: 1);
            ret.Shape.IsScalar.Should().BeTrue();
            ret.ToArray<double>()[0].Should().BeApproximately(1.2909944487358056, 0.0001d);
        }
    }
}
