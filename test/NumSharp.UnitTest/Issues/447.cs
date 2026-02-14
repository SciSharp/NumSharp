using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Issues
{
    public class Issue447
    {
        /// <summary>
        ///     https://github.com/SciSharp/NumSharp/issues/447#issuecomment-825556230
        /// </summary>
        [Test]
        public void ReproducingTest()
        {
            NDArray array3d = new NDArray(typeof(double), new Shape(new int[] {10, 50, 50}), true);
            var sumOverAxis0 = np.sum(array3d, axis: 0);
        }
    }
}
