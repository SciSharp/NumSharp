using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    [TestClass]
    public class np_outer_test
    {
        [TestMethod]
        public void Case1()
        {
            np.outer(np.ones(5), np.linspace(-2, 2, 5))
                .Should().BeOfValues(-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2)
                .And.BeShaped(5, 5);
        }
    }
}
