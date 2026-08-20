using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest
{
    /// <summary>
    /// Guards the `using` on `full = ConvolveFull(...)` inside ConvolveSame /
    /// ConvolveValid — the full convolution buffer is dead once the requested
    /// centre/valid slice has been copied out.
    /// </summary>
    /// <remarks>
    /// <c>ConvolveSame_TightLoop_DoesNotLeakWorkingSet</c> used to close this class. It is the one
    /// member of that family that had already failed CI (Ubuntu, 25 MiB, code correct), and the
    /// buffer it guarded — 1063 doubles, ~8.5 KB, 1.7 MiB over the whole loop — could not have
    /// reached its own 20 MiB threshold. Removed — see <see cref="LeakGuards"/>.
    /// </remarks>
    [TestClass]
    public class NdArray_Convolve_UsingTests
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void ConvolveSame_StillMatchesNumPy()
        {
            // np.convolve([1,2,3], [0,1,0.5], 'same') -> [1, 2.5, 4]
            var a = np.array(new double[] { 1, 2, 3 });
            var v = np.array(new double[] { 0, 1, 0.5 });
            var r = a.convolve(v, "same");

            r.Data<double>().Should().Equal(new double[] { 1, 2.5, 4 });
        }

        [TestMethod]
        public void ConvolveValid_StillMatchesNumPy()
        {
            // np.convolve([1,2,3], [0,1,0.5], 'valid') -> [2.5]
            var a = np.array(new double[] { 1, 2, 3 });
            var v = np.array(new double[] { 0, 1, 0.5 });
            var r = a.convolve(v, "valid");

            r.Data<double>().Should().Equal(new double[] { 2.5 });
        }

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// The centre/valid slice must outlive the <c>full</c> buffer it was cut from.
        /// </summary>
        /// <remarks>
        /// 'same' and 'valid' both slice their result out of <c>full</c> and then release it. A
        /// slice in NumSharp is a VIEW by default, so if either mode returned the view rather than
        /// a copy, the <c>using</c> would free the memory the caller is about to read. Both operands
        /// must survive too — they are only read from.
        /// </remarks>
        [TestMethod]
        public void ConvolveModes_ResultOutlivesTheFullBuffer()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var v = np.array(new double[] { 0, 1, 0.5 });

            var same = a.convolve(v, "same");
            var valid = a.convolve(v, "valid");

            LeakGuards.StillUsable(same, " — 'same' must copy out of `full`, not view it");
            LeakGuards.StillUsable(valid, " — 'valid' must copy out of `full`, not view it");
            LeakGuards.StillUsable(a, " — the operands are read-only inputs");
            LeakGuards.StillUsable(v);

            // Values survive the release of the buffer they came from.
            same.Data<double>().Should().Equal(new double[] { 1, 2.5, 4 });
            valid.Data<double>().Should().Equal(new double[] { 2.5 });
        }
    }
}
