using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Selection
{
    /// <summary>
    /// Correctness + working-set guards for the unified NpyIter boolean-mask
    /// axis-0 getter and setter (gather/scatter). Originally written for the
    /// hand-rolled per-iter srcSlice/destSlice loop; the assertions still hold
    /// for the allocation-free iterator path that replaced it.
    /// </summary>
    [TestClass]
    public class BooleanMasking_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        /// <summary>
        /// Axis-0 boolean mask select (the getter path that loops srcSlice +
        /// destSlice). Disposing the per-iteration view wrappers must not
        /// corrupt the result.
        /// </summary>
        [TestMethod]
        public void BooleanMask_Axis0Select_2D_StillCorrect()
        {
            var a = np.arange(20).reshape(5, 4).astype(NPTypeCode.Int32);
            var mask = new NDArray(new bool[] { true, false, true, false, true })
                .MakeGeneric<bool>();

            var picked = a[mask];

            picked.shape.Should().ContainInOrder(3L, 4L);
            for (int j = 0; j < 4; j++)
            {
                ((int)picked[0, j]).Should().Be((int)a[0, j]);
                ((int)picked[1, j]).Should().Be((int)a[2, j]);
                ((int)picked[2, j]).Should().Be((int)a[4, j]);
            }
        }

        /// <summary>
        /// Axis-0 boolean mask SET (per-iter destSlice loop). Scalar value
        /// path.
        /// </summary>
        [TestMethod]
        public void BooleanMask_Axis0Set_ScalarValue_StillCorrect()
        {
            var a = np.arange(20).reshape(5, 4).astype(NPTypeCode.Int32);
            var mask = new NDArray(new bool[] { true, false, true, false, true })
                .MakeGeneric<bool>();

            a[mask] = np.array(new[] { 99 });

            // Selected rows must be 99 across.
            for (int j = 0; j < 4; j++)
            {
                ((int)a[0, j]).Should().Be(99);
                ((int)a[2, j]).Should().Be(99);
                ((int)a[4, j]).Should().Be(99);
            }
            // Untouched rows preserved.
            ((int)a[1, 0]).Should().Be(4);
            ((int)a[3, 3]).Should().Be(15);
        }

        /// <summary>
        /// Axis-0 boolean mask SET, value broadcasting — NumPy parity. The
        /// selection result shape is (count_true,) + arr.shape[1:] = (3, 4), and
        /// the value broadcasts to it by NumPy rules:
        ///   • a 1-D length-3 value (3,) does NOT broadcast to (3, 4) → raises,
        ///     exactly like NumPy's "shape mismatch ... could not be broadcast".
        ///   • to fill one value per selected row, pass (3, 1) — it broadcasts
        ///     across the 4 columns.
        /// (The pre-unification engine silently treated a (3,) value as
        /// "scalar per selected row"; that was a divergence from NumPy.)
        /// </summary>
        [TestMethod]
        public void BooleanMask_Axis0Set_ValueBroadcast_MatchesNumPy()
        {
            var mask = new NDArray(new bool[] { true, false, true, false, true })
                .MakeGeneric<bool>();

            // (3,) value cannot broadcast to the (3, 4) selection → NumPy raises.
            var bad = np.arange(20).reshape(5, 4).astype(NPTypeCode.Int32);
            Action act = () => bad[mask] = np.array(new int[] { 100, 200, 300 });
            act.Should().Throw<IncorrectShapeException>();

            // (3, 1) broadcasts across the row → one value per selected row.
            var a = np.arange(20).reshape(5, 4).astype(NPTypeCode.Int32);
            a[mask] = np.array(new int[,] { { 100 }, { 200 }, { 300 } });

            for (int j = 0; j < 4; j++)
            {
                ((int)a[0, j]).Should().Be(100);
                ((int)a[2, j]).Should().Be(200);
                ((int)a[4, j]).Should().Be(300);
            }
            // Untouched rows still hold the originals.
            ((int)a[1, 0]).Should().Be(4);
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop of axis-0 boolean SELECTs on a 200-row array. The unified
        /// gather streams through one NpyIter pass (no per-row view wrappers),
        /// so the working set must stay near-constant across many calls.
        /// </summary>
        [TestMethod]
        public void BooleanMask_Axis0Select_TightLoop_DoesNotLeakWorkingSet()
        {
            using var a = np.arange(200 * 32).reshape(200, 32).astype(NPTypeCode.Int32);
            var maskBytes = new bool[200];
            for (int i = 0; i < 200; i++) maskBytes[i] = (i % 2) == 0;
            using var mask = new NDArray(maskBytes).MakeGeneric<bool>();

            for (int i = 0; i < 20; i++)
            {
                using var r = a[mask];
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 200; i++)
            {
                using var r = a[mask];
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // 200 outer × 200 inner = 40K view wrappers per pass. Each wrapper
            // is small but the buffer churn through the finalizer queue
            // accumulates. 30 MiB headroom covers GC variation.
            deltaMB.Should().BeLessThan(30);
        }
    }
}
