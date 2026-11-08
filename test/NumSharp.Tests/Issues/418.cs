using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #418, "help me" (https://github.com/SciSharp/NumSharp/issues/418): the reporter's
    ///     <c>(a, b) = np.meshgrid(x, y)</c> left the second grid null; since 0.80.0 <c>np.meshgrid</c> returns NumPy's tuple and
    ///     deconstructs into both grids.
    /// </summary>
    [TestClass]
    public class Issue418
    {
        /// <summary>
        ///     The answer's deconstruction gives two (2, 3) grids with NumPy 2.4.2's values; neither is null.
        /// </summary>
        [TestMethod]
        public void Meshgrid_DeconstructsIntoBothGrids()
        {
            var (xx, yy) = np.meshgrid(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 10, 20 }));

            xx.Should().NotBeNull();
            yy.Should().NotBeNull();
            xx.Should().BeShaped(2, 3).And.BeOfValues(1L, 2L, 3L, 1L, 2L, 3L);
            yy.Should().BeShaped(2, 3).And.BeOfValues(10L, 10L, 10L, 20L, 20L, 20L);
        }

        /// <summary>
        ///     The reporter's own pattern, deconstructing into existing variables and then using both: <c>scales.flatten()</c>
        ///     and <c>ratios.flatten()</c> no longer meet a null.
        /// </summary>
        [TestMethod]
        public void Meshgrid_ReportersPattern_BothGridsUsable()
        {
            NDArray scales = np.array(new double[] { 32, 64 });
            NDArray ratios = np.array(new double[] { 0.5, 1, 2 });
            (scales, ratios) = np.meshgrid(scales, ratios);

            scales.flatten().Should().BeShaped(6);
            ratios.flatten().Should().BeShaped(6).And.BeOfValues(0.5, 0.5, 1.0, 1.0, 2.0, 2.0);
        }
    }
}
