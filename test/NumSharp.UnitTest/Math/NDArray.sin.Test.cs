using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class SinTest
    {
        [TestMethod]
        public void Simple1DArray()
        {
            var nd = np.array(new double[] {0D, 0.523598775598299D,
                0.785398163397449D, 1.0471975511966D, 1.5707963267949D });

            np.sin(nd).Should().BeOfValuesApproximately(0.0001, 0, 0.5, 0.707106, 0.866025, 1);
        }
    }
}
