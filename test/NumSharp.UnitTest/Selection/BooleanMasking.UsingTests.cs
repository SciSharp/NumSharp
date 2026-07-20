using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Selection
{
    /// <summary>
    /// Correctness + lifetime guards for the unified NDIter boolean-mask
    /// axis-0 getter and setter (gather/scatter). Originally written for the
    /// hand-rolled per-iter srcSlice/destSlice loop; the assertions still hold
    /// for the allocation-free iterator path that replaced it.
    /// </summary>
    /// <remarks>
    /// <c>BooleanMask_Axis0Select_TightLoop_DoesNotLeakWorkingSet</c> used to close this class. It
    /// guarded ~2.5 MiB of churn with a 30 MiB threshold — and the iterator path it was retargeted
    /// at does not allocate the per-row wrappers it described. Removed — see <see cref="LeakGuards"/>.
    /// </remarks>
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

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// The gather must leave both the source and the mask usable, and its result must be a
        /// copy that outlives them.
        /// </summary>
        /// <remarks>
        /// Selecting twice over the same (source, mask) pair proves the first gather did not
        /// consume either operand. Disposing both afterwards then checks that the result holds its
        /// own reference to whatever buffer backs it — not that it is a copy (a correctly
        /// refcounted alias would survive too), but that nothing in the gather released a buffer
        /// the result still points at.
        /// </remarks>
        [TestMethod]
        public void BooleanMask_Axis0Select_ResultSurvivesSourceAndMaskDispose()
        {
            var a = np.arange(6 * 3).reshape(6, 3).astype(NPTypeCode.Int32);
            var maskBytes = new bool[6];
            for (int i = 0; i < 6; i++) maskBytes[i] = (i % 2) == 0;
            var mask = new NDArray(maskBytes).MakeGeneric<bool>();

            var first = a[mask];
            LeakGuards.StillUsable(a, " — the gather must not consume its source");
            LeakGuards.StillUsable(mask, " — nor its mask");

            // Second select over the same operands must still work and agree.
            var second = a[mask];
            first.shape.Should().ContainInOrder(3L, 3L);
            second.shape.Should().ContainInOrder(3L, 3L);

            // The result must still stand once both operands are gone.
            a.Dispose();
            mask.Dispose();
            LeakGuards.StillUsable(first, " — the gathered result must hold its own reference");

            // Rows 0, 2, 4 of arange(18).reshape(6,3).
            ((int)first[0, 0]).Should().Be(0);
            ((int)first[1, 0]).Should().Be(6);
            ((int)first[2, 2]).Should().Be(14);
        }
    }
}
