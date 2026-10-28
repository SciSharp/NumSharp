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

        /// <summary>
        ///     NumPy 2.4.2's np.convolve checks emptiness in Python and parses the mode only inside
        ///     <c>multiarray.correlate</c>, which then checks the depth: an empty operand wins over a bad or a null mode, and
        ///     a bad or null mode wins over a two-dimensional operand's <c>object too deep for desired array</c>.
        /// </summary>
        [TestMethod]
        public void Convolve_ChecksEmptinessBeforeTheModeAndTheModeBeforeTheDepth()
        {
            System.Action emptyA = () => np.convolve(np.zeros(new Shape(0L)), np.ones(new Shape(3L)), "bogus");
            emptyA.Should().ThrowExactly<ValueError>().Which.Message.Should().Be("a cannot be empty");
            System.Action emptyV = () => np.convolve(np.ones(new Shape(3L)), np.zeros(new Shape(0L)), null);
            emptyV.Should().ThrowExactly<ValueError>().Which.Message.Should().Be("v cannot be empty");

            System.Action badMode2D = () => np.convolve(np.zeros(new Shape(2L, 2L)), np.ones(new Shape(2L)), "bogus");
            badMode2D.Should().ThrowExactly<ValueError>().Which.Message.Should()
                .Be("mode must be one of 'valid', 'same', or 'full' (got 'bogus')");
            System.Action nullMode2D = () => np.convolve(np.zeros(new Shape(2L, 2L)), np.ones(new Shape(2L)), null);
            nullMode2D.Should().ThrowExactly<TypeError>().Which.Message.Should().Be("convolve/correlate mode not understood");
        }

        /// <summary>
        ///     NumPy 2.4.2: np.convolve's mode errors are <c>multiarray.correlate</c>'s — ValueError with the argument in
        ///     Python's <c>repr</c>, the parser's near-miss text for a letter that names a mode, and TypeError for
        ///     <c>None</c> — through np.convolve and the instance member alike.
        /// </summary>
        [TestMethod]
        public void Convolve_BadMode_IsNumPysConverterError()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var v = np.array(new double[] { 1, 0.5 });

            System.Action quote = () => np.convolve(a, v, "it's");
            quote.Should().ThrowExactly<ValueError>().Which.Message.Should()
                .Be("mode must be one of 'valid', 'same', or 'full' (got \"it's\")");
            System.Action nearMiss = () => a.convolve(v, "Full");
            nearMiss.Should().ThrowExactly<ValueError>().Which.Message.Should()
                .Be("Use one of 'valid', 'same', or 'full' for convolve/correlate mode");
            System.Action none = () => a.convolve(v, null);
            none.Should().ThrowExactly<TypeError>().Which.Message.Should().Be("convolve/correlate mode not understood");
            System.Action surrogate = () => np.convolve(a, v, "f\ud800");
            surrogate.Should().ThrowExactly<UnicodeEncodeError>().Which.Message.Should()
                .Be("'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed");
        }
    }
}
