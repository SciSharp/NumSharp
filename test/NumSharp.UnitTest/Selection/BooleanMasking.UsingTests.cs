using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Selection
{
    /// <summary>
    /// Guards the `using` on per-iter srcSlice/destSlice wrappers inside the
    /// boolean-mask axis-0 getter and setter.
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
        /// Axis-0 boolean mask SET with 1-D row-per-mask-position value
        /// (value.ndim == this.ndim - 1 branch). NumSharp interprets a 1-D
        /// value as "scalar per selected row" — value[k] broadcast across
        /// the row at mask position k.
        /// </summary>
        [TestMethod]
        public void BooleanMask_Axis0Set_OneValuePerMaskPosition_StillCorrect()
        {
            var a = np.arange(20).reshape(5, 4).astype(NPTypeCode.Int32);
            var mask = new NDArray(new bool[] { true, false, true, false, true })
                .MakeGeneric<bool>();
            // 3 mask positions → 1-D length-3 values. Each value fills a row.
            var values = np.array(new int[] { 100, 200, 300 });

            a[mask] = values;

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
        /// Tight loop of axis-0 boolean SELECTs on a 200-row array — each
        /// call loops 200 times, creating 2 view wrappers per iteration
        /// (400 per call). Working set must stay near-constant.
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
