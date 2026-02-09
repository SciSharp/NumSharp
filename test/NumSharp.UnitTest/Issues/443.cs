using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Issues
{
    public class Issue443
    {
        /// <summary>
        ///     https://github.com/SciSharp/NumSharp/issues/443#issue-825238582
        /// </summary>
        [Test]
        public void ReproducingTest()
        {
            var ones = np.ones((10, 1));
            ones.negate();
        }
    }
}
