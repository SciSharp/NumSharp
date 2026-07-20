using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    /// Guards the `using` on `comparison = this == scalar` inside NDArray.Contains.
    /// </summary>
    /// <remarks>
    /// The real guard here is <see cref="Contains_PreservesCallerInput"/>: when <c>value</c> is
    /// itself an NDArray, <c>np.asanyarray</c> hands it straight back, so a careless <c>using</c>
    /// would dispose the caller's array. <c>Contains_TightLoop_DoesNotLeakWorkingSet</c> used to sit
    /// below it and was removed — see <see cref="LeakGuards"/> for why it could not do its job.
    /// </remarks>
    [TestClass]
    public class Contains_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void Contains_PresentValue_True()
        {
            var arr = np.array(new[] { 1, 2, 3, 4, 5 });
            arr.Contains(3).Should().BeTrue();
        }

        [TestMethod]
        public void Contains_AbsentValue_False()
        {
            var arr = np.array(new[] { 1, 2, 3, 4, 5 });
            arr.Contains(10).Should().BeFalse();
        }

        [TestMethod]
        public void Contains_2D_PresentValue_True()
        {
            var arr = np.arange(20).reshape(4, 5);
            arr.Contains(13).Should().BeTrue();
        }

        [TestMethod]
        public void Contains_PreservesCallerInput()
        {
            // When `value` is itself an NDArray, np.asanyarray returns it as-is.
            // The `using` on `comparison` (the equality result) must NOT dispose
            // either the caller's `arr` or `value`.
            var arr = np.array(new[] { 1, 2, 3, 4, 5 });
            var query = np.array(new[] { 3 });
            arr.Contains((object)query).Should().BeTrue();

            // Both the array and the query must remain usable after Contains.
            LeakGuards.StillUsable(arr, " — Contains must not dispose the array it searches");
            LeakGuards.StillUsable(query, " — asanyarray returns the query as-is; `using` must not free it");
            ((int)arr[2]).Should().Be(3);
            ((int)query[0]).Should().Be(3);

            // And still answers correctly on a second pass over the same pair.
            arr.Contains((object)query).Should().BeTrue();
        }
    }
}
