using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Reproductions for GitHub issue #448, "Debug.Assert(...) causes tests to stop the entire process": a boolean-mask
    ///     assignment followed by reading the array back used to hit a <c>Debug.Assert</c> in a Debug build, which ended the
    ///     whole test host instead of failing one test.
    /// </summary>
    /// <remarks>
    ///     Both tests carried <c>[OpenBugs]</c>, and the class had no <c>[TestClass]</c> from the TUnit → MSTest migration
    ///     (ac020336, 2026-04-15) until 2026-10-05, so MSTest never ran them. The 2026-10-05 run of the whole OpenBugs
    ///     category added the attribute and found both passing in Debug and Release builds, so no assert fires on these
    ///     paths any more and the tags were removed. The issue itself stays open on GitHub: it is about asserts in general,
    ///     and the VSTest host now turns a failed <c>Debug.Assert</c> into a <c>DebugAssertException</c> instead of ending
    ///     the process.
    /// </remarks>
    [TestClass]
    public class Issue448
    {
        /// <summary>
        ///     <c>nd[nd &lt; 3] = -2</c> writes through the boolean mask: <c>[[1, 2, 3], [4, 5, 6]]</c> becomes
        ///     <c>[[-2, -2, 3], [4, 5, 6]]</c>, and the comparison that builds the mask reads the values unchanged.
        /// </summary>
        [TestMethod]
        public void ReproducingTest1()
        {
            NDArray nd = new double[,] {{1, 2, 3}, {4, 5, 6}};
            (nd < 3).Should().BeOfValues(true, true, false, false, false, false);

            nd[(NDArray)(nd < 3)] = -2;
            nd.Should().BeOfValues(-2, -2, 3, 4, 5, 6);
        }

        /// <summary>
        ///     After the same boolean-mask write, iterating the array element by element (<c>AsElements&lt;double&gt;()</c>)
        ///     walks all six elements without an assert or an exception.
        /// </summary>
        [TestMethod]
        public void ReproducingTest2()
        {
            NDArray nd = new double[,] {{1, 2, 3}, {4, 5, 6}};
            (nd < 3).Should().BeOfValues(true, true, false, false, false, false);

            nd[(NDArray)(nd < 3)] = -2;
            var values = new int[] {-2, -2, 3, 4, 5, 6};

            var iter = nd.AsElements<double>();
            var next = iter.MoveNext;
            var hasnext = iter.HasNext;
            for (int i = 0; i < values.Length; i++)
            {
                next();
            }
        }
    }
}
