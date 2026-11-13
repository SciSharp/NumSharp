using System.Linq;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #462, "How to use the repo to convert some Python code?"
    ///     (https://github.com/SciSharp/NumSharp/issues/462): with an image's pixel bytes in an <c>NDArray</c>, the Python line
    ///     <c>np.average(img_yuv[:,:,2])</c> ports one to one through the string slice indexer.
    /// </summary>
    [TestClass]
    public class Issue462
    {
        /// <summary>
        ///     The answer's code minus OpenCvSharp: the bytes of a 2 x 3 three-channel image (0 .. 17) become an (H, W, 3)
        ///     uint8 array through <c>np.frombuffer</c> + <c>reshape</c>, and <c>np.average(yuv[":, :, 2"])</c> is the mean of
        ///     channel 2 (2, 5, 8, 11, 14, 17), 9.5 as in NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void AverageOfAChannel_PortsOneToOne()
        {
            int rows = 2, cols = 3;
            var bytes = Enumerable.Range(0, rows * cols * 3).Select(i => (byte)i).ToArray();
            var yuv = np.frombuffer(bytes, np.uint8).reshape(rows, cols, 3);

            var averageV = np.average(yuv[":, :, 2"]);

            averageV.ndim.Should().Be(0);
            averageV.GetDouble().Should().Be(9.5);
        }
    }
}
