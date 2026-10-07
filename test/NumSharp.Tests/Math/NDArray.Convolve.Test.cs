using System.Linq;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// Test concolve with standard example from
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayConvolveTest
    {
        [TestMethod]
        public void ConvoleFull()
        {
            var series1 = np.array(new double[] {1, 2, 3});
            var series2 = np.array(new double[] {0, 1, 0.5});

            var series3 = series1.convolve(series2);

            double[] expectedResult = new double[] {0, 1, 2.5, 4, 1.5};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data<double>(), expectedResult));
        }

        [TestMethod]
        public void ConvoleValid()
        {
            var series1 = np.array(new double[] {1, 2, 3});
            var series2 = np.array(new double[] {0, 1, 0.5});

            var series3 = series1.convolve(series2, "valid");

            double[] expectedResult = new double[] {2.5};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data<double>(), expectedResult));
        }

        [TestMethod]
        public void ConvoleSame()
        {
            var series1 = np.array(new double[] {1, 2, 3});
            var series2 = np.array(new double[] {0, 1, 0.5});

            var series3 = series1.convolve(series2, "same");

            double[] expectedResult = new double[] {1, 2.5, 4};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data<double>(), expectedResult));
        }

        /// <summary>
        ///     NumPy 2.4.2's np.convolve checks emptiness on the operands AS GIVEN, before it swaps the shorter one to the
        ///     second slot: <c>np.convolve([], [1, 1, 1])</c> is <c>a cannot be empty</c> (the swap-first order reported
        ///     <c>v</c>), <c>np.convolve([1, 1, 1], [])</c> is <c>v cannot be empty</c>.
        /// </summary>
        [TestMethod]
        public void Convolve_ChecksEmptinessBeforeTheLengthSwap()
        {
            System.Action emptyA = () => np.convolve(np.zeros(new Shape(0L)), np.ones(new Shape(3L)));
            emptyA.Should().Throw<ValueError>().WithMessage("a cannot be empty");
            System.Action emptyV = () => np.convolve(np.ones(new Shape(3L)), np.zeros(new Shape(0L)));
            emptyV.Should().Throw<ValueError>().WithMessage("v cannot be empty");
        }
    }
}
