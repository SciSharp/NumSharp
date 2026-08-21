namespace NumSharp.Tests.RandomSampling
{
    [TestClass]
    public class NpRandomNormalTest
    {
        [TestMethod]
        public void NormalDistributionTest()
        {
            // https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
            double mu = 0; // mean
            double sigma = 0.1; // standard deviation
            var s = np.random.normal(mu, sigma, new Shape(10, 100));

            var mean = np.mean(s);
            Assert.IsTrue(System.Math.Abs(mu - mean.Data<double>()[0]) < 0.01);

            var std = s.std();
            // Assert.IsTrue(Math.Abs(sigma - std.Data<double>()[0] )  < 0.01);

            Assert.IsTrue(s.shape[0] == 10);
            Assert.IsTrue(s.shape[1] == 100);
        }
    }
}
