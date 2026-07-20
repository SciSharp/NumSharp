using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Guards the <c>using</c> on the <c>np.isclose</c> intermediate inside
    /// <c>DefaultEngine.AllClose</c>. The intermediate is a bool array the
    /// shape of broadcast(a, b), dead once <c>np.all</c> has collapsed it.
    /// </summary>
    /// <remarks>
    /// This class used to close with <c>AllClose_TightLoop_DoesNotLeakWorkingSet</c>. It was
    /// removed: measured, it passed with the leak reintroduced and failed CI on Ubuntu with the
    /// code correct. See <see cref="LeakGuards"/> for the numbers — and do not add another.
    /// </remarks>
    [TestClass]
    public class np_allclose_using_test : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void AllClose_TrueCase_AfterRefactor()
        {
            // NumPy: np.allclose([1e10, 1e-8], [1.00001e10, 1e-9]) -> True
            np.allclose(new[] { 1e10, 1e-8 }, new[] { 1.00001e10, 1e-9 })
                .Should().BeTrue();
        }

        [TestMethod]
        public void AllClose_FalseCase_AfterRefactor()
        {
            // NumPy: np.allclose([1e10, 1e-7], [1.00001e10, 1e-8]) -> False
            np.allclose(new[] { 1e10, 1e-7 }, new[] { 1.00001e10, 1e-8 })
                .Should().BeFalse();
        }

        [TestMethod]
        public void AllClose_EqualNan_AfterRefactor()
        {
            // equal_nan=True: NaN==NaN by special-case branch in IsClose.
            np.allclose(new[] { 1.0, np.nan }, new[] { 1.0, np.nan }, equal_nan: true)
                .Should().BeTrue();
            np.allclose(new[] { 1.0, np.nan }, new[] { 1.0, np.nan })
                .Should().BeFalse();
        }

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// The <c>using</c> must release the closeness array and nothing else.
        /// </summary>
        /// <remarks>
        /// <c>np.isclose</c> broadcasts its operands, and a broadcast of an already-correct shape
        /// can legitimately hand back a view onto the input rather than a fresh buffer. Should the
        /// intermediate ever become such an alias, <c>using</c> would free the caller's memory from
        /// under it. Both operands must therefore still be usable — and still hold their values —
        /// after the call, including on the equal-shape path where aliasing is most likely.
        /// </remarks>
        [TestMethod]
        public void AllClose_DoesNotDisposeItsOperands()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            var b = np.array(new[] { 1.0, 2.0, 3.0 });

            np.allclose(a, b).Should().BeTrue();

            LeakGuards.StillUsable(a, " — np.isclose's result must not alias operand a");
            LeakGuards.StillUsable(b, " — np.isclose's result must not alias operand b");
            a.Data<double>().Should().Equal(new[] { 1.0, 2.0, 3.0 });
            b.Data<double>().Should().Equal(new[] { 1.0, 2.0, 3.0 });

            // Still answering correctly on a second pass over the same operands.
            np.allclose(a, b).Should().BeTrue();
        }

        /// <summary>
        /// The broadcasting path (shape (3,1) against (3,)) allocates a differently-shaped
        /// intermediate; releasing it must likewise leave both operands intact.
        /// </summary>
        [TestMethod]
        public void AllClose_Broadcast_DoesNotDisposeItsOperands()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0 }).reshape(3, 1);
            var b = np.array(new[] { 1.0, 2.0, 3.0 });

            np.allclose(a, b).Should().BeFalse();  // 3x3 comparison, off-diagonal differs

            LeakGuards.StillUsable(a);
            LeakGuards.StillUsable(b);
            ((double)a[0, 0]).Should().Be(1.0);
            ((double)b[2]).Should().Be(3.0);
        }
    }
}
