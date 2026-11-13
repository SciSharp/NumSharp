namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #372, "Clustering Example" (https://github.com/SciSharp/NumSharp/issues/372) as
    ///     not planned: no k-means example is planned, but the API gap the thread noted is fixed - since 0.80.0
    ///     <c>np.random.choice(a, size, replace, p)</c> takes NumPy's arguments and draws NumPy's stream.
    /// </summary>
    /// <remarks>
    ///     <c>[DoNotParallelize]</c>: the tests seed the GLOBAL <c>np.random</c> and expect NumPy's stream (TestParallelization's
    ///     rule).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class Issue372
    {
        /// <summary>
        ///     <c>choice(5, 3, replace: true, p: [0.1, 0.2, 0.3, 0.2, 0.2])</c> after <c>seed(0)</c> is NumPy 2.4.2's
        ///     <c>[2, 3, 3]</c>, and the same draw without replacement is <c>[2, 3, 1]</c>.
        /// </summary>
        [TestMethod]
        public void Choice_TakesNumPysArguments_AndDrawsNumPysStream()
        {
            var p = np.array(new[] { 0.1, 0.2, 0.3, 0.2, 0.2 });

            np.random.seed(0);
            np.random.choice(5, new Shape(3), replace: true, p: p).ToArray<long>().Should().Equal(2, 3, 3);

            np.random.seed(0);
            np.random.choice(5, new Shape(3), replace: false, p: p).ToArray<long>().Should().Equal(2, 3, 1);
        }
    }
}
