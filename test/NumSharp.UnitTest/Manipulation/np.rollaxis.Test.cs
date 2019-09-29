using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_rollaxis_Test
    {
        [TestMethod]
        public void Case1()
        {
            var a = np.ones((3, 4, 5, 6));
            np.rollaxis(a, 3, 1).Should().BeShaped(3, 6, 4, 5);
            np.rollaxis(a, 2).Should().BeShaped(5, 3, 4, 6);
            np.rollaxis(a, 1, 4).Should().BeShaped(3, 5, 6, 4);
        }
    }
}
