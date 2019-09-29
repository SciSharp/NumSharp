using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_moveaxis_Test
    {
        [TestMethod]
        public void Case1()
        {
            var x = np.zeros((3, 4, 5));
            np.moveaxis(x, 0, -1).Should().BeShaped(4, 5, 3);
            np.moveaxis(x, -1, 0).Should().BeShaped(5, 4, 3);
        }        

        [TestMethod]
        public void Case2()
        {
            var x = np.zeros((3, 4, 5));
            np.moveaxis(x, new[] {0, 1}, new[] {-1, -2}).Should().BeShaped(5,4,3);
            np.moveaxis(x, new[] {0, 1, 2}, new[] {-1, -2, -3}).Should().BeShaped(5,4,3);
        }
    }
}
