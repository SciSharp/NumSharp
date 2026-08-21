using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Math
{
    [TestClass]
    public class LogTest
    {
        [TestMethod]
        public void Case1()
        {
            var np1 = np.array(new double[] {1, System.Math.E, System.Math.E * System.Math.E, 0}); // .MakeGeneric<double>();
            np.log(np1).Should().BeOfValues(0, 1, 2, double.NegativeInfinity);
        }
    }
}
