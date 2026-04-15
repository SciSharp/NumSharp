using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Issues
{
    [TestClass]
    public class Issue443
    {
        /// <summary>
        ///     https://github.com/SciSharp/NumSharp/issues/443#issue-825238582
        /// </summary>
        [TestMethod]
        public void ReproducingTest()
        {
            var ones = np.ones((10, 1));
            ones.negate();
        }
    }
}
