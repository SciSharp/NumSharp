using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Guards the `using` around `flat = m.flat` inside np.eye — flat is
    /// purely a write iterator for the diagonal and never returned.
    /// </summary>
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

        // --------------------------- leak guard ---------------------------

        [TestMethod]
        public void Eye_TightLoop_DoesNotLeakWorkingSet()
        {
            for (int i = 0; i < 20; i++)
                _ = np.eye(100);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 1000; i++)
            {
                using var e = np.eye(100);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // 1000 × 100×100 doubles = ~80 MiB raw, but each call has only
            // the result + the flat wrapper. Disposing inputs via using
            // keeps steady state near zero. 20 MiB headroom.
            deltaMB.Should().BeLessThan(20);
        }
    }
}
