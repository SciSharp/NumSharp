using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Statistics
{
    [TestClass]
    public class NdArrayMeanTest
    {

        [TestMethod]
        public void Case1_Elementwise_keepdims()
        {
            var np1 = np.array(new double[] { 1, 2, 3, 4, 5, 6 }).reshape(3, 2);
            var mean = np.mean(np1, keepdims: true);
            mean.Shape.Should().Be(new Shape(1, 1));
            mean.GetValue(0, 0).Should().BeEquivalentTo(3.5);
        }

        [TestMethod]
        public void Case0_Scalar()
        {
            var np1 = NDArray.Scalar(1d);
            var mean = np.mean(np1);
            mean.Shape.IsScalar.Should().BeTrue();
            mean.GetValue(0).Should().BeEquivalentTo(1d);
        }

        [TestMethod]
        public void Case1_Axis0()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4}).reshape(2, 2);
            var mean = np.mean(np1, 0);
            Assert.IsTrue(Enumerable.SequenceEqual(mean.Data<double>(), new double[] {2, 3}));
        }

        [TestMethod]
        public void Case1_Axis1()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4}).reshape(2, 2);
            var mean = np.mean(np1, 1);
            Assert.IsTrue(Enumerable.SequenceEqual(mean.Data<double>(), new double[] {1.5, 3.5}));
        }

        [TestMethod]
        public void Case1_Axis_minus1()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4}).reshape(2, 2);
            var mean = np.mean(np1, -1);
            mean.Shape.dimensions.Should().ContainInOrder(2);
            Assert.IsTrue(Enumerable.SequenceEqual(mean.Data<double>(), new double[] {1.5, 3.5}));
        }

        [TestMethod]
        public void Case1_Elementwise()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4, 5, 6}).reshape(3, 2);
            var mean = np.mean(np1);
            mean.Shape.IsScalar.Should().BeTrue();
            mean.GetValue(0).Should().BeEquivalentTo(3.5);
        }
    }
}
