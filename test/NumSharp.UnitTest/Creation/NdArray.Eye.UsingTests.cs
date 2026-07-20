using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Guards the `using` around `flat = m.flat` inside np.eye — flat is
    /// purely a write iterator for the diagonal and never returned.
    /// </summary>
    /// <remarks>
    /// <c>Eye_TightLoop_DoesNotLeakWorkingSet</c> used to live here and was removed; the wrapper it
    /// guarded is ~100 B against a 20 MiB threshold, so it could not have failed. See
    /// <see cref="LeakGuards"/>.
    /// </remarks>
    [TestClass]
    public class NdArray_Eye_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void Eye_Square_StillCorrect()
        {
            var e = np.eye(3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    ((double)e[i, j]).Should().Be(i == j ? 1.0 : 0.0);
        }

        [TestMethod]
        public void Eye_Offset_StillCorrect()
        {
            // np.eye(4, k=1) has ones on the super-diagonal.
            var e = np.eye(4, k: 1);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    ((double)e[i, j]).Should().Be((j == i + 1) ? 1.0 : 0.0);
        }

        [TestMethod]
        public void Eye_Rectangular_StillCorrect()
        {
            var e = np.eye(3, 5);  // 3x5, ones on main diagonal
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 5; j++)
                    ((double)e[i, j]).Should().Be(i == j ? 1.0 : 0.0);
        }

        [TestMethod]
        public void Eye_Int_Dtype_StillCorrect()
        {
            var e = np.eye(3, dtype: typeof(int));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    ((int)e[i, j]).Should().Be(i == j ? 1 : 0);
        }

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// Disposing the <c>flat</c> write-iterator must not release the matrix it writes into.
        /// </summary>
        /// <remarks>
        /// This is the one lifetime hazard in np.eye: <c>flat</c> is a VIEW over <c>m</c>, sharing
        /// its buffer through the ARC refcount, and <c>m</c> is what np.eye returns. If the
        /// <c>using</c> on the view were to drop the buffer's last reference, np.eye would hand back
        /// an array whose memory has already been freed — reachable, correctly shaped, and pointing
        /// at released pages. The refcount assertion catches that where reading values may not.
        /// </remarks>
        [TestMethod]
        public void Eye_ResultOutlivesItsFlatWriteIterator()
        {
            var e = np.eye(4, k: 1);

            LeakGuards.StillUsable(e, " — the `flat` view must not take np.eye's result with it");

            // The diagonal written THROUGH the disposed view is still readable and correct.
            ((double)e[0, 1]).Should().Be(1.0);
            ((double)e[2, 3]).Should().Be(1.0);
            ((double)e[3, 0]).Should().Be(0.0);
        }
    }
}
