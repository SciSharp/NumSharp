using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NPY_ITER_REDUCTION_AXIS encoding.
    /// NumPy: common.h:347 (macro), nditer_constr.c:1439 (decoder npyiter_get_op_axis).
    ///
    /// Semantics: additive encoding axis + (1 &lt;&lt; 30). Values >= (1 &lt;&lt; 30) - 1 are
    /// treated as reduction-axis-flagged entries in op_axes[iop][idim]. When decoded,
    /// the original axis is recovered and the is_reduction flag is set.
    ///
    /// Parity with NumPy:
    ///   NPY_ITER_REDUCTION_AXIS(0)  == 0x40000000
    ///   NPY_ITER_REDUCTION_AXIS(-1) == 0x3FFFFFFF
    ///   NPY_ITER_REDUCTION_AXIS(5)  == 0x40000005
    /// </summary>
    [TestClass]
    public class NpyIterReductionAxisEncodingTests
    {
        // ============================================================
        // Encoding / decoding primitives
        // ============================================================

        [TestMethod]
        public void ReductionAxis_Offset_IsCorrect()
        {
            Assert.AreEqual(1 << 30, NpyIterConstants.REDUCTION_AXIS_OFFSET);
            Assert.AreEqual(0x40000000, NpyIterConstants.REDUCTION_AXIS_OFFSET);
        }

        [TestMethod]
        public void ReductionAxis_EncodesPositiveAxis()
        {
            Assert.AreEqual(0x40000000, NpyIterUtils.ReductionAxis(0));
            Assert.AreEqual(0x40000001, NpyIterUtils.ReductionAxis(1));
            Assert.AreEqual(0x40000005, NpyIterUtils.ReductionAxis(5));
        }

        [TestMethod]
        public void ReductionAxis_EncodesNegativeOneAsForcedBroadcast()
        {
            // NPY_ITER_REDUCTION_AXIS(-1) = 0x3FFFFFFF
            Assert.AreEqual(0x3FFFFFFF, NpyIterUtils.ReductionAxis(-1));
        }

        [TestMethod]
        public void GetOpAxis_DecodesPlainAxis()
        {
            int axis = NpyIterUtils.GetOpAxis(3, out bool isReduction);
            Assert.AreEqual(3, axis);
            Assert.IsFalse(isReduction);
        }

        [TestMethod]
        public void GetOpAxis_DecodesMinusOne()
        {
            int axis = NpyIterUtils.GetOpAxis(-1, out bool isReduction);
            Assert.AreEqual(-1, axis);
            Assert.IsFalse(isReduction);
        }

        [TestMethod]
        public void GetOpAxis_DecodesReductionFlaggedAxis()
        {
            int axis = NpyIterUtils.GetOpAxis(NpyIterUtils.ReductionAxis(2), out bool isReduction);
            Assert.AreEqual(2, axis);
            Assert.IsTrue(isReduction);
        }

        [TestMethod]
        public void GetOpAxis_DecodesReductionFlaggedMinusOne()
        {
            // NPY_ITER_REDUCTION_AXIS(-1) — threshold case
            int encoded = NpyIterUtils.ReductionAxis(-1);
            int axis = NpyIterUtils.GetOpAxis(encoded, out bool isReduction);
            Assert.AreEqual(-1, axis);
            Assert.IsTrue(isReduction);
        }

        [TestMethod]
        public void GetOpAxis_RoundTrip()
        {
            for (int i = -1; i < 10; i++)
            {
                int encoded = NpyIterUtils.ReductionAxis(i);
                int decoded = NpyIterUtils.GetOpAxis(encoded, out bool isRed);
                Assert.IsTrue(isRed, $"axis={i}");
                Assert.AreEqual(i, decoded, $"axis={i}");
            }
        }

        // ============================================================
        // Integration: ApplyOpAxes correctly handles explicit reduction
        // ============================================================

        [TestMethod]
        public void ExplicitReduction_WithReduceOk_Succeeds()
        {
            // Setup: sum along axis 0 using explicit reduction axis encoding.
            // x shape (3,4), y shape (4,), op_axes=[[0,1], [REDUCTION_AXIS(-1),0]]
            // The REDUCTION_AXIS(-1) entry says "output doesn't have this axis — reduce it"
            var x = np.arange(12).reshape(3, 4).astype(np.int32);
            var y = np.zeros(new int[] { 4 }, np.int32);

            var opAxes = new[]
            {
                new[] { 0, 1 },
                new[] { NpyIterUtils.ReductionAxis(-1), 0 },
            };

            using var it = NpyIterRef.AdvancedNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                opDtypes: null,
                opAxesNDim: 2,
                opAxes: opAxes);

            // Should succeed and mark the iterator as a reduction
            Assert.IsTrue(it.IsReduction);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ExplicitReduction_WithoutReduceOk_Throws()
        {
            var x = np.arange(12).reshape(3, 4).astype(np.int32);
            var y = np.zeros(new int[] { 4 }, np.int32);

            var opAxes = new[]
            {
                new[] { 0, 1 },
                new[] { NpyIterUtils.ReductionAxis(-1), 0 },
            };

            using var it = NpyIterRef.AdvancedNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.BUFFERED,  // No REDUCE_OK!
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                opDtypes: null,
                opAxesNDim: 2,
                opAxes: opAxes);
        }

        [TestMethod]
        public void PlainAxis_NoReductionFlag_NotReduction()
        {
            // Plain op_axes (no encoding) should behave as before
            var x = np.arange(6).reshape(2, 3).astype(np.int32);
            var y = np.zeros(new int[] { 2, 3 }, np.int32);

            using var it = NpyIterRef.AdvancedNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY },
                opDtypes: null,
                opAxesNDim: 2,
                opAxes: new[] { new[] { 0, 1 }, new[] { 0, 1 } });

            Assert.IsFalse(it.IsReduction);
        }
    }
}
