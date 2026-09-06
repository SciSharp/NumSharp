using System.Linq;

namespace NumSharp.Tests.Creation
{
    [TestClass]
    public class np_zeros_like_Test
    {
        [TestMethod]
        public void SimpleInt1D()
        {
            // create same-shaped zeros from ones
            var np1 = np.zeros_like(np.ones(new Shape(5)));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 5);
        }

        [TestMethod]
        public void SimpleInt2D()
        {
            // create same-shaped zeros from ones
            var np1 = np.zeros_like(np.ones(new Shape(5, 5)));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 25);
        }

        [TestMethod]
        public void SimpleDouble3D()
        {
            // create same-shaped zeros from ones
            var np1 = np.zeros_like(np.ones(new Shape(5, 5, 5)));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 125);
        }
    }
}
